#!/usr/bin/env python3
"""Dependency-free checks for the typed-decision JSONL contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLITS = {"train", "dev", "calibration", "test", "ood"}


def probabilities_are_valid(value: Any, candidates: list[str]) -> bool:
    if not isinstance(value, dict) or set(value) != set(candidates):
        return False
    if any(isinstance(value[name], bool) or not isinstance(value[name], (int, float)) for name in candidates):
        return False
    return all(0 <= float(value[name]) <= 1 for name in candidates) and abs(sum(float(value[name]) for name in candidates) - 1.0) <= 1e-6


def validate(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("id", "state_id", "family_id", "split", "state", "questions"):
        if field not in row:
            errors.append(f"missing:{field}")
    if row.get("split") not in SPLITS:
        errors.append("invalid:split")
    if not isinstance(row.get("state"), dict) or not isinstance(row.get("questions"), dict) or not row.get("questions"):
        errors.append("invalid:state_or_questions")
        return errors
    gold = row.get("gold", {})
    for qid, question in row["questions"].items():
        kind = question.get("type")
        if kind not in {"choice", "boolean", "score"}:
            errors.append(f"{qid}:invalid_type")
            continue
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                errors.append(f"{qid}:invalid_choice_criteria")
            elif qid in gold and gold[qid] not in criteria:
                errors.append(f"{qid}:gold_not_candidate")
        elif kind == "boolean" and qid in gold and type(gold[qid]) is not bool:
            errors.append(f"{qid}:gold_not_boolean")
        elif kind == "score" and not isinstance(criteria, list):
            errors.append(f"{qid}:invalid_score_criteria")
    for field in ("gold", "gold_label_kind"):
        if not isinstance(row.get(field, {}), dict):
            errors.append(f"invalid:{field}")
        elif set(row.get(field, {})) - set(row["questions"]):
            errors.append(f"{field}:unknown_question")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_split: dict[str, dict[str, int]] = {}
    all_errors: list[dict[str, Any]] = []
    seen_state: dict[str, str] = {}
    rows = 0
    for split in sorted(SPLITS):
        path = args.input_dir / f"{split}.jsonl"
        count = 0
        errors = 0
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                rows += 1
                count += 1
                try:
                    row = json.loads(line)
                    row_errors = validate(row)
                    state_id = row.get("state_id")
                    if state_id in seen_state and seen_state[state_id] != split:
                        row_errors.append("state_crosses_splits")
                    elif state_id:
                        seen_state[state_id] = split
                except (json.JSONDecodeError, TypeError) as exc:
                    row_errors = [f"json:{exc}"]
                if row_errors:
                    errors += 1
                    all_errors.append({"split": split, "line": line_number, "errors": row_errors})
        by_split[split] = {"rows": count, "rows_with_errors": errors}
    report = {"schema_version": "browser-intent-contract-validation-v1", "input_dir": str(args.input_dir), "rows": rows, "by_split": by_split, "errors": all_errors, "passed": not all_errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": rows, "errors": len(all_errors), "passed": not all_errors}, ensure_ascii=False))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
