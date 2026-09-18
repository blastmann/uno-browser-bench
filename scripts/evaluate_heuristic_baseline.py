#!/usr/bin/env python3
"""Score the local heuristic predictor on the synthetic Browser Observation set."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_predictor.server import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    by_length: dict[str, list[int]] = defaultdict(list)
    by_domain: dict[str, list[int]] = defaultdict(list)
    predictions = Counter()
    for row in rows:
        output = predict(row["observation"])
        choice = output["intent"]
        correct = int(choice == row["trajectory_gold"]["intent"])
        by_length[str(row["trajectory_length"])].append(correct)
        by_domain[row["domain"]].append(correct)
        predictions[choice] += 1
    report = {
        "schema_version": "browser-intent-heuristic-baseline-v1",
        "model": "heuristic-v0",
        "rows": len(rows),
        "accuracy": sum(sum(values) for values in by_length.values()) / len(rows) if rows else None,
        "predicted_intent_counts": dict(predictions),
        "by_length": {key: {"n": len(values), "accuracy": sum(values) / len(values)} for key, values in sorted(by_length.items(), key=lambda item: int(item[0]))},
        "by_domain": {key: {"n": len(values), "accuracy": sum(values) / len(values)} for key, values in sorted(by_domain.items())},
        "note": "This is a deterministic protocol baseline, not a trained model.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "accuracy": report["accuracy"], "by_length": report["by_length"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
