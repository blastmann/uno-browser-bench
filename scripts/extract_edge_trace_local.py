#!/usr/bin/env python3
"""Extract privacy-preserving browser traces from a local Edge History DB.

This script is intentionally local-only. It opens the Chromium History SQLite
database read-only and emits category/action sequences, never URLs, titles,
hostnames, query strings, paths, timestamps, or hashes derived from them.

The labels are *observed next page categories*, not ground-truth user intent.
They are useful for measuring a personal transition prior and for constructing
an explicitly weakly-labelled trace benchmark. Human labels are still needed
for claims about semantic intent accuracy.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlparse

try:
    from .analyze_edge_history import category
except ImportError:  # Running this file directly from the scripts directory.
    from analyze_edge_history import category


EPOCH_OFFSET_US = 11644473600000000
DEFAULT_GAP_MINUTES = 30
DEFAULT_MAX_EVENTS = 256
HORIZONS = (5, 10, 20, 50, 100, 200)

TRANSITION_NAMES = {
    0: "link",
    1: "typed",
    2: "bookmark",
    3: "subframe",
    4: "manual_subframe",
    5: "generated",
    6: "auto_toplevel",
    7: "form_submit",
    8: "reload",
    9: "keyword",
    10: "keyword_generated",
}

WEAK_INTENT_BY_CATEGORY = {
    "email": "communication",
    "search": "information_search",
    "github_code": "development",
    "docs_dev": "development",
    "ecommerce": "shopping",
    "social": "social",
    "news": "news",
    "other": "other",
}


def transition_name(value: int | None) -> str:
    """Return the low-byte Chromium transition type without raw metadata."""

    if value is None:
        return "unknown"
    return TRANSITION_NAMES.get(int(value) & 0xFF, "other")


def weak_intent(page_type: str) -> str:
    """Map a page category to a deliberately weak intent proxy."""

    return WEAK_INTENT_BY_CATEGORY.get(page_type, "other")


def _history_uri(history: Path) -> str:
    # quote() keeps spaces and non-ASCII profile paths valid in SQLite URIs.
    return f"file:{quote(str(history.resolve()), safe='/\\:')}?mode=ro&immutable=1"


def read_sessions(
    history: Path,
    *,
    gap_minutes: int = DEFAULT_GAP_MINUTES,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    """Read and sessionize visits while retaining only safe coarse fields."""

    sessions: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_time: int | None = None
    raw_rows_scanned = 0
    ignored_rows = 0
    truncated_sessions = 0
    gap_us = gap_minutes * 60 * 1_000_000

    # The database is opened immutable/read-only. No SQLite journal or lock is
    # created next to the user's browser profile.
    with closing(sqlite3.connect(_history_uri(history), uri=True)) as db:
        query = (
            "SELECT visits.visit_time, visits.transition, urls.url "
            "FROM visits JOIN urls ON urls.id = visits.url "
            "WHERE urls.url IS NOT NULL "
            "ORDER BY visits.visit_time, visits.id"
        )
        for raw_time, raw_transition, raw_url in db.execute(query):
            raw_rows_scanned += 1
            parsed = urlparse(raw_url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                ignored_rows += 1
                continue

            if previous_time is not None and raw_time - previous_time > gap_us:
                if current:
                    sessions.append(current)
                current = []
            previous_time = raw_time

            if len(current) < max_events:
                page_type = category(parsed.hostname)
                current.append(
                    {
                        "step": len(current),
                        "action": transition_name(raw_transition),
                        "page_type": page_type,
                    }
                )
            else:
                truncated_sessions += 1

        if current:
            sessions.append(current)

    stats = {
        "raw_rows_scanned": raw_rows_scanned,
        "ignored_non_http_rows": ignored_rows,
        "sessions_found": len(sessions),
        "sessions_truncated_at_max_events": truncated_sessions,
    }
    return sessions, stats


def split_for_session(index: int, total: int) -> str:
    """Use a chronological session split to avoid prefix leakage."""

    if total <= 1:
        return "train"
    fraction = index / total
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "dev"
    return "test"


def build_prefix_rows(
    sessions: Iterable[list[dict[str, Any]]],
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand sessions into next-category prediction prefixes."""

    session_list = list(sessions)
    eligible = [events for events in session_list if len(events) >= 2]
    rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    horizon_counts: Counter[str] = Counter()

    for session_index, events in enumerate(eligible):
        trace_id = f"edge-local-{session_index:05d}"
        split = split_for_session(session_index, len(eligible))
        for event in events:
            category_counts[event["page_type"]] += 1
            action_counts[event["action"]] += 1
        for prefix_length in range(1, len(events)):
            label = events[prefix_length]["page_type"]
            prefix = events[:prefix_length]
            row = {
                "schema_version": "edge-trace-prefix-v1",
                "trace_id": trace_id,
                "session_index": session_index,
                "split": split,
                "prefix_length": prefix_length,
                "events": prefix,
                "label": {
                    "kind": "observed_next_page_category",
                    "page_type": label,
                    "weak_intent_proxy": weak_intent(label),
                },
            }
            rows.append(row)
            length_counts[str(prefix_length)] += 1
            if prefix_length in horizons:
                horizon_counts[str(prefix_length)] += 1

    audit = {
        "schema_version": "edge-trace-audit-v1",
        "sessions_total": len(session_list),
        "sessions_eligible_min_2_events": len(eligible),
        "prefix_rows": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)),
        "sessions_by_split": dict(
            Counter(
                split_for_session(index, len(eligible))
                for index in range(len(eligible))
            )
        ),
        "page_type_counts": dict(sorted(category_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "prefix_length_counts": dict(sorted(length_counts.items(), key=lambda item: int(item[0]))),
        "available_horizons": {
            str(horizon): horizon_counts.get(str(horizon), 0) for horizon in horizons
        },
        "labels": {
            "primary": "observed_next_page_category",
            "secondary": "weak_intent_proxy",
            "ground_truth_semantic_intent": False,
        },
        "privacy": {
            "raw_urls_written": False,
            "raw_titles_written": False,
            "hostnames_written": False,
            "query_strings_written": False,
            "paths_written": False,
            "timestamps_written": False,
            "url_hashes_written": False,
            "network_calls": 0,
        },
    }
    return rows, audit


def write_jsonl(rows: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--session-gap-minutes", type=int, default=DEFAULT_GAP_MINUTES)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    args = parser.parse_args()
    if args.session_gap_minutes <= 0 or args.max_events < 2:
        parser.error("--session-gap-minutes must be positive and --max-events must be >= 2")

    sessions, scan_stats = read_sessions(
        args.history,
        gap_minutes=args.session_gap_minutes,
        max_events=args.max_events,
    )
    rows, audit = build_prefix_rows(sessions)
    audit.update(scan_stats)
    audit["source"] = "local Edge History SQLite, read-only, immutable"
    audit["history_file"] = args.history.name
    audit["session_gap_minutes"] = args.session_gap_minutes
    audit["max_events_per_session"] = args.max_events
    audit["note"] = (
        "This is a local weak-label trace set. Observed next page category is "
        "not proof of the user's semantic intent."
    )

    row_count = write_jsonl(rows, args.output)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit_output": str(args.audit_output),
                "prefix_rows_written": row_count,
                "sessions": audit["sessions_eligible_min_2_events"],
                "available_horizons": audit["available_horizons"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
