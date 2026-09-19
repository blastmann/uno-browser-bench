#!/usr/bin/env python3
"""Prepare local Mind2Web test actions for typed candidate selection.

This follows the released Decider Mind2Web loader: one positive element and
five negative candidates, with the task, website, operation, and recent
action representations as context. Raw HTML is deliberately discarded.
The extracted JSONL stays under work/ and is ignored by Git.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SEED = 0


def elem_desc(candidate: dict[str, Any]) -> str:
    try:
        attributes = json.loads(candidate.get("attributes", "{}"))
    except Exception:
        attributes = {}
    bits = [candidate.get("tag", "")]
    for key in ("aria_label", "aria-label", "title", "alt", "placeholder", "name", "value", "type", "role", "id", "class"):
        if attributes.get(key):
            bits.append(f"{key}={str(attributes[key])[:40]}")
    return " ".join(bits)[:120]


def prepare_split(files: list[Path], split: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            previous: list[str] = []
            for step_index, (action, action_repr) in enumerate(zip(record["actions"], record["action_reprs"])):
                positives = action.get("pos_candidates") or []
                if positives:
                    positive = elem_desc(positives[0])
                    negatives = [elem_desc(candidate) for candidate in action.get("neg_candidates", [])]
                    negatives = [candidate for candidate in negatives if candidate != positive]
                    random.Random(SEED + len(rows)).shuffle(negatives)
                    descriptions = [positive] + negatives[:5]
                    random.Random(SEED + len(rows) + 1).shuffle(descriptions)
                    if len(descriptions) < 2:
                        previous.append(action_repr)
                        continue
                    criteria = {f"c{index}": text for index, text in enumerate(descriptions)}
                    gold_id = next(key for key, text in criteria.items() if text == positive)
                    value = action.get("operation", {}).get("value") or ""
                    context = (
                        f"Website: {record['website']} ({record['domain']}).\n"
                        f"Task: {record['confirmed_task']}\n"
                        f"Actions so far: {'; '.join(previous[-4:]) if previous else 'none'}\n"
                        f"Next operation: {action['operation']['op']}{' ' + repr(value) if value else ''}"
                    )
                    rows.append(
                        {
                            "id": f"{record['annotation_id']}:step:{step_index}",
                            "split": split,
                            "task_id": record["annotation_id"],
                            "step_index": step_index,
                            "website": record["website"],
                            "domain": record["domain"],
                            "operation": action["operation"]["op"],
                            "state": context,
                            "questions": {
                                "target": {
                                    "type": "choice",
                                    "instructions": "Which page element should the next operation target?",
                                    "criteria": criteria,
                                }
                            },
                            "gold": gold_id,
                            "candidate_count": len(criteria),
                        }
                    )
                previous.append(action_repr)
    random.Random(SEED).shuffle(rows)
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int, default=1500)
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error("--limit-per-split must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"schema_version": "mind2web-typed-test-v1", "seed": SEED, "splits": {}}
    for split in ("test_task", "test_website", "test_domain"):
        split_dir = args.test_root / split
        files = sorted(split_dir.glob("*.json"))
        rows = prepare_split(files, split, args.limit_per_split)
        output = args.output_dir / f"{split}.jsonl"
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        task_ids = {row["task_id"] for row in rows}
        summary["splits"][split] = {
            "source_files": len(files),
            "rows": len(rows),
            "tasks": len(task_ids),
            "websites": len({row["website"] for row in rows}),
            "domains": len({row["domain"] for row in rows}),
            "candidate_count_distribution": {
                str(k): sum(1 for row in rows if row["candidate_count"] == k) for k in range(2, 7)
            },
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
