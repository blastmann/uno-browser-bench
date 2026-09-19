#!/usr/bin/env python3
"""Audit decision-contract split validity without emitting raw observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "calibration", "test", "ood")


def load_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for split in SPLITS:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            errors.append({"split": split, "error": "missing_file"})
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row["_split_file"] = split
                rows.append(row)
            except json.JSONDecodeError as exc:
                errors.append({"split": split, "line": line_number, "error": str(exc)})
    return rows, errors


def signature(row: dict[str, Any]) -> str:
    payload = {"state": row.get("state"), "questions": row.get("questions")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit(input_dir: Path) -> dict[str, Any]:
    rows, parse_errors = load_rows(input_dir)
    required = ("id", "state_id", "family_id", "split", "state", "questions", "gold")
    missing_fields = [
        {"id": row.get("id"), "fields": [field for field in required if field not in row]}
        for row in rows
        if any(field not in row for field in required)
    ]
    ids = Counter(str(row.get("state_id")) for row in rows)
    duplicate_state_ids = sorted(key for key, count in ids.items() if count > 1)
    families_by_split: dict[str, set[str]] = defaultdict(set)
    labels_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    lengths_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    signature_to_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    split_field_mismatch = []
    for row in rows:
        split = row.get("_split_file")
        families_by_split[split].add(str(row.get("family_id")))
        labels_by_split[split][str((row.get("gold") or {}).get("intent", "unknown"))] += 1
        lengths_by_split[split][str((row.get("metadata") or {}).get("requested_prefix_length", "unknown"))] += 1
        signature_to_rows[signature(row)].append(row)
        if row.get("split") != split:
            split_field_mismatch.append(row.get("state_id"))

    family_overlap = []
    split_names = list(families_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(families_by_split[left] & families_by_split[right])
            if overlap:
                family_overlap.append({"left": left, "right": right, "families": overlap})

    conflicting_duplicate_labels = []
    duplicate_input_groups = 0
    for input_sig, group in signature_to_rows.items():
        if len(group) <= 1:
            continue
        duplicate_input_groups += 1
        labels = sorted({str((row.get("gold") or {}).get("intent", "unknown")) for row in group})
        split_set = sorted({row.get("_split_file") for row in group})
        if len(labels) > 1:
            conflicting_duplicate_labels.append({"signature_prefix": input_sig[:12], "labels": labels, "splits": split_set})

    non_train = {split: len(labels_by_split.get(split, {})) for split in SPLITS if split != "train"}
    report = {
        "schema_version": "browser-intent-split-audit-v2",
        "input_dir": str(input_dir),
        "rows": len(rows),
        "parse_errors": parse_errors,
        "missing_required_fields": missing_fields,
        "duplicate_state_ids": duplicate_state_ids,
        "split_field_mismatch": split_field_mismatch,
        "duplicate_input_groups": duplicate_input_groups,
        "conflicting_duplicate_labels": conflicting_duplicate_labels,
        "family_overlap": family_overlap,
        "by_split": {
            split: {
                "rows": sum(1 for row in rows if row.get("_split_file") == split),
                "families": sorted(families_by_split.get(split, set())),
                "family_count": len(families_by_split.get(split, set())),
                "label_count": len(labels_by_split.get(split, {})),
                "labels": dict(sorted(labels_by_split.get(split, {}).items())),
                "lengths": dict(sorted(lengths_by_split.get(split, {}).items())),
            }
            for split in SPLITS
        },
        "held_out_label_counts": non_train,
        "passed": not (
            parse_errors
            or missing_fields
            or duplicate_state_ids
            or split_field_mismatch
            or family_overlap
            or conflicting_duplicate_labels
            or any(non_train[split] < 2 for split in non_train)
        ),
        "interpretation": (
            "Family disjointness prevents template leakage. This audit does not "
            "turn synthetic completed-task labels into personal semantic intent."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": report["rows"], "passed": report["passed"]}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
