#!/usr/bin/env python3
"""Audit Browser Observation traces before training.

The report is deliberately dependency-free and treats the input as data. It
does not infer that repeated events are valid long trajectories; it surfaces
the filler ratio and label consistency for review.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REQUIRED = ("id", "observation", "events", "trajectory_gold", "trajectory_length")
FILLER_ACTIONS = {"scroll", "wait", "mousemove", "mouseover", "idle"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:16]


def normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def core_length(events: list[dict[str, Any]]) -> int:
    """Find the first long filler tail, conservatively.

    This is a diagnostic heuristic, not a label rewrite. A tail is considered
    filler only after three consecutive filler events; otherwise the trace is
    left untouched and the report says that no filler tail was detected.
    """
    run = 0
    for i, event in enumerate(events):
        if normalized(event.get("action")) in FILLER_ACTIONS:
            run += 1
            if run >= 3:
                return i - 2
        else:
            run = 0
    return len(events)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append({"line": line_number, "error": str(exc)})
    return rows, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, parse_errors = load_jsonl(args.input)
    missing: list[dict[str, Any]] = []
    invalid_lengths: list[dict[str, Any]] = []
    duplicate_inputs: collections.Counter[str] = collections.Counter()
    input_labels: dict[str, set[str]] = collections.defaultdict(set)
    domain_counts: collections.Counter[str] = collections.Counter()
    length_counts: collections.Counter[str] = collections.Counter()
    domain_length: collections.Counter[str] = collections.Counter()
    intent_counts: collections.Counter[str] = collections.Counter()
    filler_ratios: list[float] = []
    filler_tail_rows: list[dict[str, Any]] = []
    leakage_hits: list[dict[str, Any]] = []
    family_counts: collections.Counter[str] = collections.Counter()

    for row in rows:
        missing_keys = [key for key in REQUIRED if key not in row]
        if missing_keys:
            missing.append({"id": row.get("id"), "keys": missing_keys})
            continue

        observation = row["observation"]
        events = row["events"]
        gold = row["trajectory_gold"]
        declared_length = row["trajectory_length"]
        actual_length = len(events) if isinstance(events, list) else -1
        if not isinstance(events, list) or actual_length != declared_length:
            invalid_lengths.append({"id": row.get("id"), "declared": declared_length, "actual": actual_length})
            continue

        domain = str(row.get("domain", "unknown"))
        intent = str(gold.get("intent", "unknown")) if isinstance(gold, dict) else "unknown"
        input_signature = digest({"observation": observation, "events": events})
        duplicate_inputs[input_signature] += 1
        input_labels[input_signature].add(intent)
        domain_counts[domain] += 1
        length_counts[str(declared_length)] += 1
        domain_length[f"{domain}|{declared_length}"] += 1
        intent_counts[intent] += 1
        family = f"{domain}|{intent}"
        family_counts[family] += 1

        detected_core = core_length(events)
        filler = max(0, len(events) - detected_core)
        ratio = filler / len(events) if events else 0.0
        filler_ratios.append(ratio)
        if filler >= 3:
            filler_tail_rows.append({"id": row.get("id"), "length": len(events), "core_length": detected_core, "filler_events": filler, "filler_ratio": ratio})

        text_blob = normalized({"url": observation.get("url"), "title": observation.get("title")})
        if intent != "unknown" and normalized(intent).replace("_", " ") in text_blob:
            leakage_hits.append({"id": row.get("id"), "field": "observation.url/title", "intent": intent})

    conflicting_labels = [
        {"input_signature": signature, "labels": sorted(labels), "count": duplicate_inputs[signature]}
        for signature, labels in input_labels.items()
        if len(labels) > 1
    ]
    duplicate_groups = [
        {"input_signature": signature, "count": count, "labels": sorted(input_labels[signature])}
        for signature, count in duplicate_inputs.items()
        if count > 1
    ]

    report = {
        "schema_version": "browser-intent-audit-v1",
        "input": str(args.input),
        "rows": len(rows),
        "parse_errors": parse_errors,
        "missing_required_fields": missing,
        "invalid_event_lengths": invalid_lengths,
        "counts": {
            "domains": dict(sorted(domain_counts.items())),
            "trajectory_lengths": dict(sorted(length_counts.items(), key=lambda item: int(item[0]))),
            "domain_by_length": dict(sorted(domain_length.items())),
            "intents": dict(sorted(intent_counts.items())),
            "families": dict(sorted(family_counts.items())),
        },
        "duplicate_input_groups": duplicate_groups,
        "conflicting_duplicate_labels": conflicting_labels,
        "filler_tail": {
            "rows_with_detected_tail": len(filler_tail_rows),
            "mean_ratio": sum(filler_ratios) / len(filler_ratios) if filler_ratios else 0.0,
            "max_ratio": max(filler_ratios, default=0.0),
            "examples": filler_tail_rows[:100],
        },
        "possible_label_leakage": leakage_hits,
        "review_notes": [
            "A filler tail is a diagnostic heuristic and does not rewrite labels.",
            "Repeated input signatures across splits must be checked after split generation.",
            "This report cannot infer the user's latent intent from a completed trajectory.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "errors": len(parse_errors), "duplicate_groups": len(duplicate_groups), "filler_tail_rows": len(filler_tail_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
