#!/usr/bin/env python3
"""Fit temperature on calibration predictions and report held-out curves.

The prediction JSON contains only typed probabilities and state ids. Gold
metadata is joined from the local decision contract. No Browser Observation
text is copied into the report.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "dev", "calibration", "test", "ood")
EPS = 1e-12


def load_gold(input_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            result[row["state_id"]] = {
                "split": split,
                "gold": row["gold"]["intent"],
                "family_id": row.get("family_id"),
                "length": int(row.get("metadata", {}).get("requested_prefix_length", 0)),
            }
    return result


def load_prediction_rows(predictions: Path, gold: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for state in payload.get("states", []):
        state_id = state["id"]
        meta = gold.get(state_id)
        answer = state.get("answers", {}).get("intent", {})
        probabilities = answer.get("probabilities")
        if not meta or not isinstance(probabilities, dict) or not probabilities:
            continue
        candidates = sorted(probabilities)
        probs = [max(float(probabilities[name]), 0.0) for name in candidates]
        total = math.fsum(probs)
        if total <= 0 or not math.isfinite(total):
            continue
        probs = [value / total for value in probs]
        if meta["gold"] not in candidates:
            continue
        rows.append(
            {
                "id": state_id,
                "split": meta["split"],
                "family_id": meta["family_id"],
                "length": meta["length"],
                "gold": meta["gold"],
                "gold_index": candidates.index(meta["gold"]),
                "candidates": candidates,
                "probs": probs,
            }
        )
    return rows


def logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def scaled_probs(probs: list[float], temperature: float) -> list[float]:
    logits = [math.log(max(value, EPS)) / temperature for value in probs]
    normalizer = logsumexp(logits)
    return [math.exp(value - normalizer) for value in logits]


def nll(rows: list[dict[str, Any]], temperature: float) -> float:
    if not rows:
        return 0.0
    losses = []
    for row in rows:
        probs = scaled_probs(row["probs"], temperature)
        losses.append(-math.log(max(probs[row["gold_index"]], EPS)))
    return math.fsum(losses) / len(losses)


def fit_temperature(rows: list[dict[str, Any]]) -> float:
    """Fit one scalar T by deterministic coarse-to-fine NLL minimization."""

    if not rows:
        return 1.0
    best_t = 1.0
    best_loss = float("inf")
    # Log-spaced search is stable even when the model is extremely confident.
    for index in range(241):
        log_t = -3.0 + index * 6.0 / 240.0
        temperature = math.exp(log_t)
        loss = nll(rows, temperature)
        if loss < best_loss:
            best_t, best_loss = temperature, loss
    # Refine around the best grid point without requiring scipy.
    step = 6.0 / 240.0
    center = math.log(best_t)
    for _ in range(4):
        candidates = [center - step, center, center + step]
        winner = min(candidates, key=lambda value: nll(rows, math.exp(value)))
        center = winner
        step /= 4.0
    return math.exp(center)


def ece(rows: list[dict[str, Any]], bins: int = 10) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for bucket_index in range(bins):
        low = bucket_index / bins
        high = (bucket_index + 1) / bins
        bucket = [
            row
            for row in rows
            if low <= row["confidence"] < high
            or (bucket_index == bins - 1 and low <= row["confidence"] <= high)
        ]
        if bucket:
            confidence = math.fsum(row["confidence"] for row in bucket) / len(bucket)
            accuracy = math.fsum(row["correct"] for row in bucket) / len(bucket)
            total += len(bucket) / len(rows) * abs(confidence - accuracy)
    return total


def macro_f1(rows: list[dict[str, Any]]) -> float:
    labels = sorted({row["gold"] for row in rows} | {row["pred"] for row in rows})
    values = []
    for label in labels:
        tp = sum(row["gold"] == label and row["pred"] == label for row in rows)
        fp = sum(row["gold"] != label and row["pred"] == label for row in rows)
        fn = sum(row["gold"] == label and row["pred"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return math.fsum(values) / len(values) if values else 0.0


def materialize(rows: list[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        probs = scaled_probs(row["probs"], temperature)
        pred_index = max(range(len(probs)), key=probs.__getitem__)
        item = dict(row)
        item["calibrated_probs"] = probs
        item["pred"] = row["candidates"][pred_index]
        item["confidence"] = probs[pred_index]
        item["correct"] = int(pred_index == row["gold_index"])
        result.append(item)
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": None, "nll": None, "brier": None, "ece": None, "macro_f1": None}
    loss = []
    brier = []
    for row in rows:
        probs = row["calibrated_probs"]
        loss.append(-math.log(max(probs[row["gold_index"]], EPS)))
        brier.append(math.fsum((p - float(i == row["gold_index"])) ** 2 for i, p in enumerate(probs)))
    ordered = sorted(rows, key=lambda row: row["confidence"], reverse=True)
    risk_coverage = {}
    for fraction in (0.25, 0.5, 0.75, 1.0):
        count = max(1, math.ceil(len(ordered) * fraction))
        selected = ordered[:count]
        risk_coverage[str(fraction)] = {
            "n": count,
            "selective_accuracy": math.fsum(row["correct"] for row in selected) / count,
            "threshold": selected[-1]["confidence"],
        }
    return {
        "n": len(rows),
        "accuracy": math.fsum(row["correct"] for row in rows) / len(rows),
        "macro_f1": macro_f1(rows),
        "nll": math.fsum(loss) / len(loss),
        "brier": math.fsum(brier) / len(brier),
        "ece": ece(rows),
        "mean_confidence": math.fsum(row["confidence"] for row in rows) / len(rows),
        "risk_coverage": risk_coverage,
    }


def by_length(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["length"]].append(row)
    return {str(length): summarize(values) for length, values in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-split", default="calibration")
    args = parser.parse_args()

    gold = load_gold(args.gold_dir)
    rows = load_prediction_rows(args.predictions, gold)
    calibration = [row for row in rows if row["split"] == args.calibration_split]
    temperature = fit_temperature(calibration)
    calibrated = materialize(rows, temperature)
    report = {
        "schema_version": "browser-intent-calibration-curve-v1",
        "predictions": str(args.predictions),
        "gold_dir": str(args.gold_dir),
        "calibration_split": args.calibration_split,
        "rows": len(calibrated),
        "rows_by_split": dict(Counter(row["split"] for row in calibrated)),
        "families_by_split": {
            split: len({row["family_id"] for row in calibrated if row["split"] == split})
            for split in sorted({row["split"] for row in calibrated})
        },
        "temperature": {
            "fitted": True,
            "value": temperature,
            "calibration_rows": len(calibration),
            "uncalibrated_nll": nll(calibration, 1.0),
            "calibrated_nll": nll(calibration, temperature),
        },
        "overall": summarize(calibrated),
        "by_split": {
            split: summarize([row for row in calibrated if row["split"] == split])
            for split in sorted({row["split"] for row in calibrated})
        },
        "long_trajectory_curve": {
            split: by_length([row for row in calibrated if row["split"] == split])
            for split in ("calibration", "test", "ood")
        },
        "limitations": [
            "Calibration is fitted on the independent synthetic calibration split only.",
            "The labels are completed-task labels from the synthetic benchmark, not personal latent intent.",
            "The prediction probabilities are rounded by the Decider output interface.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "temperature": temperature, "rows": len(calibrated)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
