#!/usr/bin/env python3
"""Evaluate typed Choice predictions with accuracy and calibration metrics."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ece(rows: list[dict[str, Any]], bins: int = 10) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [r for r in rows if lo <= r["confidence"] < hi or (b == bins - 1 and r["confidence"] <= hi)]
        if bucket:
            total += len(bucket) / len(rows) * abs(mean([r["confidence"] for r in bucket]) - mean([r["correct"] for r in bucket]))
    return total


def scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": None, "macro_f1": None, "nll": None, "brier": None, "ece": None}
    labels = sorted({r["gold"] for r in rows} | {r["pred"] for r in rows})
    f1s = []
    for label in labels:
        tp = sum(r["pred"] == label and r["gold"] == label for r in rows)
        fp = sum(r["pred"] == label and r["gold"] != label for r in rows)
        fn = sum(r["pred"] != label and r["gold"] == label for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "n": len(rows),
        "accuracy": mean([r["correct"] for r in rows]),
        "macro_f1": mean(f1s),
        "nll": mean([-math.log(max(r["gold_prob"], 1e-12)) for r in rows]),
        "brier": mean([sum((p - (1.0 if i == r["gold_index"] else 0.0)) ** 2 for i, p in enumerate(r["probs"])) for r in rows]),
        "ece": ece(rows),
        "mean_confidence": mean([r["confidence"] for r in rows]),
    }


def risk_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: r["confidence"], reverse=True)
    out = {}
    for coverage in (0.25, 0.5, 0.75, 1.0):
        n = max(1, math.ceil(len(ordered) * coverage))
        chosen = ordered[:n]
        out[str(coverage)] = {"n": n, "coverage": n / len(ordered), "selective_accuracy": mean([r["correct"] for r in chosen]), "threshold": chosen[-1]["confidence"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            probs = [float(value) for value in raw["student_probs"]]
            candidates = raw["candidate_ids"]
            gold_index = int(raw["gold_index"])
            pred_index = max(range(len(probs)), key=probs.__getitem__)
            rows.append({
                "id": raw.get("id"),
                "split": raw.get("split"),
                "family_id": raw.get("family_id"),
                "length": str(raw.get("state_id", "").rsplit(":", 1)[-1]),
                "gold": candidates[gold_index],
                "pred": candidates[pred_index],
                "gold_index": gold_index,
                "pred_index": pred_index,
                "gold_prob": probs[gold_index],
                "confidence": probs[pred_index],
                "correct": int(pred_index == gold_index),
                "probs": probs,
            })
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append({"line": line_number, "error": str(exc)})
    by_length: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[row["length"]].append(row)
        by_family[row["family_id"]].append(row)
    report = {
        "schema_version": "browser-intent-decision-evaluation-v1",
        "input": str(args.input),
        "rows": len(rows),
        "parse_errors": errors,
        "overall": scores(rows),
        "risk_coverage": risk_coverage(rows),
        "by_length": {key: scores(value) for key, value in sorted(by_length.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0])},
        "by_family": {key: scores(value) for key, value in sorted(by_family.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "errors": len(errors), "overall": report["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
