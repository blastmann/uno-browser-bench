#!/usr/bin/env python3
"""Evaluate NanoJev inference JSON against the decision-contract gold rows."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold: dict[str, dict] = {}
    for path in args.gold_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gold[row["id"] + ":intent"] = row
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    rows = []
    for state in payload.get("states", []):
        state_id = state["id"]
        row = gold.get(state_id + ":intent")
        answer = state.get("answers", {}).get("intent", {})
        if not row or not answer:
            continue
        probs = answer.get("probabilities", {})
        gold_label = row["gold"]["intent"]
        choice = answer.get("choice")
        confidence = float(probs.get(choice, 0.0)) if choice is not None else 0.0
        rows.append({
            "id": state_id,
            "length": int(state_id.rsplit(":", 1)[-1]),
            "family_id": row.get("family_id"),
            "gold": gold_label,
            "pred": choice,
            "correct": int(choice == gold_label),
            "confidence": confidence,
        })
    by_length: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_length[row["length"]].append(row)
    report = {
        "schema_version": "browser-intent-typed-inference-evaluation-v1",
        "predictions": str(args.predictions),
        "rows": len(rows),
        "overall": {"accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None, "mean_confidence": sum(r["confidence"] for r in rows) / len(rows) if rows else None},
        "by_length": {str(length): {"n": len(values), "accuracy": sum(r["correct"] for r in values) / len(values), "mean_confidence": sum(r["confidence"] for r in values) / len(values)} for length, values in sorted(by_length.items())},
        "rows_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "overall": report["overall"], "by_length": report["by_length"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
