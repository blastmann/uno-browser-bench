#!/usr/bin/env python3
"""Read Edge History in SQLite read-only mode and emit privacy-preserving aggregates.

The script never writes a copy of History and never emits raw URLs, titles,
query strings, paths, or page contents. The output is intended to stay under
results/personal/ and is ignored by git.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


EPOCH_OFFSET_US = 11644473600000000


def category(host: str) -> str:
    host = host.lower()
    rules = {
        "github_code": ("github.", "gitlab.", "bitbucket."),
        "search": ("google.", "bing.", "duckduckgo.", "baidu.", "search."),
        "email": ("mail.", "gmail.", "outlook.", "proton.me", "qq.com"),
        "ecommerce": ("amazon.", "ebay.", "taobao.", "jd.com", "shop.", "store."),
        "social": ("reddit.", "x.com", "twitter.", "facebook.", "linkedin.", "weibo."),
        "news": ("news.", "nytimes.", "bbc.", "theguardian.", "medium."),
        "docs_dev": ("docs.", "developer.", "pypi.", "npmjs.", "stackoverflow."),
    }
    for name, fragments in rules.items():
        if any(fragment in host for fragment in fragments):
            return name
    return "other"


def edge_time(value: int | None) -> str | None:
    if not value:
        return None
    seconds = (value - EPOCH_OFFSET_US) / 1_000_000
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uri = f"file:{args.history.resolve().as_posix()}?mode=ro&immutable=1"
    counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    category_transitions: Counter[str] = Counter()
    days: Counter[str] = Counter()
    total = 0
    first = None
    last = None
    with sqlite3.connect(uri, uri=True) as db:
        query = "SELECT url, last_visit_time, visit_count, typed_count FROM urls WHERE url IS NOT NULL"
        for url, raw_time, visit_count, typed_count in db.execute(query):
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            host_category = category(parsed.hostname)
            counts[host_category] += max(int(visit_count or 1), 1)
            total += 1
            stamp = edge_time(raw_time)
            if stamp:
                day = stamp[:10]
                days[day] += 1
                first = stamp if first is None or stamp < first else first
                last = stamp if last is None or stamp > last else last
            if int(typed_count or 0) > 0:
                transitions["typed_navigation"] += int(typed_count)
        previous_category = None
        previous_time = None
        sessions = 0
        transition_query = "SELECT urls.url, visits.visit_time FROM visits JOIN urls ON urls.id = visits.url ORDER BY visits.visit_time"
        for url, raw_time in db.execute(transition_query):
            parsed = urlparse(url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            current_category = category(parsed.hostname)
            if previous_time is None or raw_time - previous_time > 30 * 60 * 1_000_000:
                sessions += 1
                previous_category = None
            if previous_category and previous_category != current_category:
                category_transitions[f"{previous_category}->{current_category}"] += 1
            previous_category = current_category
            previous_time = raw_time
    report = {
        "schema_version": "edge-history-aggregate-v1",
        "source": "local Edge History SQLite, read-only",
        "raw_rows_scanned": total,
        "visit_counts_by_category": dict(sorted(counts.items())),
        "typed_navigation_count": transitions["typed_navigation"],
        "sessions_30min_gap": sessions,
        "category_transitions": dict(category_transitions.most_common(30)),
        "active_days": len(days),
        "rows_by_day": dict(sorted(days.items())),
        "first_visit_utc": first,
        "last_visit_utc": last,
        "privacy": {
            "raw_urls_written": False,
            "raw_titles_written": False,
            "query_strings_written": False,
            "paths_written": False,
            "network_calls": 0,
        },
        "note": "Category counts are local telemetry and must not be published without an explicit review.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "raw_rows_scanned": total, "categories": dict(counts), "active_days": len(days)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
