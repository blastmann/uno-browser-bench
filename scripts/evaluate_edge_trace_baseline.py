#!/usr/bin/env python3
"""Evaluate a local weak-label next-category baseline on extracted traces."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def top_items(counter: Counter[str], limit: int = 3) -> list[str]:
    return [item for item, _ in counter.most_common(limit)]


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in rows if row.get("split") == "train"]
    held_out = [row for row in rows if row.get("split") in {"dev", "test"}]
    by_category_action: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in train:
        events = row.get("events") or []
        if not events:
            continue
        last = events[-1]
        key = (last["page_type"], last["action"])
        label = row["label"]["page_type"]
        by_category_action[key][label] += 1
        by_category[last["page_type"]][label] += 1
        global_counts[label] += 1

    def prediction(row: dict[str, Any]) -> list[str]:
        last = row["events"][-1]
        exact = by_category_action.get((last["page_type"], last["action"]))
        if exact:
            return top_items(exact)
        category_counter = by_category.get(last["page_type"])
        if category_counter:
            return top_items(category_counter)
        return top_items(global_counts)

    by_horizon: dict[str, dict[str, float | int]] = {}
    for horizon in (5, 10, 20, 50, 100, 200):
        subset = [row for row in held_out if row.get("prefix_length") == horizon]
        if not subset:
            continue
        top1 = 0
        top3 = 0
        for row in subset:
            pred = prediction(row)
            gold = row["label"]["page_type"]
            top1 += int(bool(pred) and pred[0] == gold)
            top3 += int(gold in pred[:3])
        by_horizon[str(horizon)] = {
            "examples": len(subset),
            "top1_accuracy": round(top1 / len(subset), 6),
            "top3_accuracy": round(top3 / len(subset), 6),
        }

    overall_top1 = 0
    overall_top3 = 0
    confusion: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in held_out:
        pred = prediction(row)
        gold = row["label"]["page_type"]
        if pred and pred[0] == gold:
            overall_top1 += 1
        if gold in pred[:3]:
            overall_top3 += 1
        confusion[gold][pred[0] if pred else "unknown"] += 1

    label_set = sorted(set(confusion) | {pred for values in confusion.values() for pred in values})
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    non_other_total = 0
    non_other_correct = 0
    for label in label_set:
        true_positive = confusion[label].get(label, 0)
        predicted_total = sum(values.get(label, 0) for values in confusion.values())
        gold_total = sum(confusion[label].values())
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": gold_total,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        f1_values.append(f1)
    for row in held_out:
        gold = row["label"]["page_type"]
        if gold == "other":
            continue
        non_other_total += 1
        pred = prediction(row)
        non_other_correct += int(bool(pred) and pred[0] == gold)

    result = {
        "schema_version": "edge-trace-baseline-v1",
        "label_kind": "observed_next_page_category",
        "train_rows": len(train),
        "held_out_rows": len(held_out),
        "overall": {
            "top1_accuracy": round(overall_top1 / len(held_out), 6) if held_out else None,
            "top3_accuracy": round(overall_top3 / len(held_out), 6) if held_out else None,
            "non_other_examples": non_other_total,
            "non_other_top1_accuracy": (
                round(non_other_correct / non_other_total, 6) if non_other_total else None
            ),
            "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else None,
        },
        "by_prefix_length": by_horizon,
        "confusion": {gold: dict(preds) for gold, preds in sorted(confusion.items())},
        "per_label": per_label,
        "train_label_distribution": dict(global_counts.most_common()),
        "note": (
            "This evaluates transition prediction from coarse categories. It does "
            "not measure semantic intent recognition or user satisfaction."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
