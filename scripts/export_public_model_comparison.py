#!/usr/bin/env python3
"""Export model-comparison results without publishing raw browser observations.

The exporter intentionally keeps only aggregate metrics and per-example outcome
metadata. It removes raw states, HTML, URLs, candidate IDs/text, predictions,
and source identifiers. Hashes are scoped by dataset and are truncated to 16
hex characters so that public files cannot be joined accidentally across
benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def scoped_hash(scope: str, value: Any) -> str:
    payload = f"public-model-comparison-v1::{scope}::{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def finite_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def gold_probability(row: dict[str, Any]) -> float | None:
    probabilities = row.get("probabilities")
    gold = row.get("gold")
    if not isinstance(probabilities, dict) or not isinstance(gold, str):
        return None
    return finite_number(probabilities.get(gold))


def mind2web_rows(path: Path, provider: str) -> Iterable[dict[str, Any]]:
    payload = load_json(path)
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        yield {
            "dataset": "mind2web_test_domain",
            "provider": provider,
            "example_hash": scoped_hash("mind2web-example", row.get("id")),
            "task_hash": scoped_hash("mind2web-task", row.get("annotation_id")),
            "coarse_domain": row.get("domain"),
            "correct": bool(row.get("correct")),
            "confidence": finite_number(row.get("confidence")),
            "gold_probability": gold_probability(row),
            "latency_ms": finite_number(row.get("latency_ms")),
        }


def uno_rows(path: Path, provider: str, split: str) -> Iterable[dict[str, Any]]:
    payload = load_json(path)
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        yield {
            "dataset": "uno_trace_v3",
            "split": split,
            "provider": provider,
            "example_hash": scoped_hash(f"uno-example-{split}", row.get("id")),
            "trajectory_hash": scoped_hash(f"uno-trajectory-{split}", row.get("source_group_id")),
            "family_hash": scoped_hash(f"uno-family-{split}", row.get("family_id")),
            "coarse_domain": row.get("domain"),
            "prefix_length": row.get("requested_prefix_length"),
            "correct": bool(row.get("correct")),
            "confidence": finite_number(row.get("confidence")),
            "gold_probability": gold_probability(row),
            "latency_ms": finite_number(row.get("latency_ms")),
        }


def selected_mind2web_aggregate(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    providers = {}
    for provider, summary in payload.get("providers", {}).items():
        providers[provider] = {
            "n_valid": summary.get("n_valid"),
            "accuracy": summary.get("accuracy"),
            "mean_confidence": summary.get("mean_confidence"),
            "ece_10": summary.get("ece_10"),
            "brier": summary.get("brier"),
            "latency_ms": summary.get("latency_ms"),
            "task_metrics": summary.get("task_metrics"),
            "by_domain": summary.get("by_domain"),
        }
    paired = {}
    for name, value in payload.get("paired", {}).items():
        paired[name] = {
            key: value.get(key)
            for key in ("a_correct_b_wrong", "b_correct_a_wrong", "both_correct", "both_wrong",
                        "same_choice_rate", "mcnemar_exact_two_sided_p")
            if key in value
        }
    return {
        "dataset": "Mind2Web test_domain",
        "evaluated_common": payload.get("evaluated_common"),
        "valid_with_ground_truth_candidate": payload.get("valid_with_ground_truth_candidate"),
        "excluded_examples": payload.get("excluded_no_positive_candidate", payload.get("excluded_no_cleaned_positive_candidate")),
        "providers": providers,
        "paired": paired,
        "privacy": {"raw_states": False, "raw_urls": False, "raw_candidates": False, "raw_predictions": False},
    }


def selected_uno_aggregate(paths: dict[tuple[str, str], Path]) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset": "Uno trace-v3", "splits": {}, "privacy": {
        "raw_states": False, "raw_urls": False, "raw_candidates": False, "raw_predictions": False,
    }}
    for (split, provider), path in sorted(paths.items()):
        payload = load_json(path)
        summary = payload.get("summary", {})
        split_payload = result["splits"].setdefault(split, {})
        split_payload[provider] = {
            "overall": summary.get("overall"),
            "by_prefix": summary.get("by_prefix"),
            "trajectory_strict": summary.get("trajectory_strict"),
        }
    return result


def write_json(path: Path, value: Any) -> None:
    def json_safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): json_safe(value) for key, value in item.items()}
        if isinstance(item, list):
            return [json_safe(value) for value in item]
        if isinstance(item, Path):
            return str(item)
        return item

    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mind2web-comparison", type=Path, required=True)
    parser.add_argument("--mind2web-decider", type=Path, required=True)
    parser.add_argument("--mind2web-jev", type=Path, required=True)
    parser.add_argument("--mind2web-nanojev", type=Path, required=True)
    parser.add_argument("--uno", action="append", nargs=3, metavar=("SPLIT", "PROVIDER", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    m2w_aggregate = selected_mind2web_aggregate(args.mind2web_comparison)
    uno_paths: dict[tuple[str, str], Path] = {}
    all_rows: list[dict[str, Any]] = []
    all_rows.extend(mind2web_rows(args.mind2web_decider, "decider-2b"))
    all_rows.extend(mind2web_rows(args.mind2web_jev, "jev"))
    all_rows.extend(mind2web_rows(args.mind2web_nanojev, "nanojev"))
    for split, provider, raw_path in args.uno:
        path = Path(raw_path)
        uno_paths[(split, provider)] = path
        all_rows.extend(uno_rows(path, provider, split))

    aggregate = {
        "schema_version": "public-model-comparison-v1",
        "generated_on": str(date.today()),
        "mind2web": m2w_aggregate,
        "uno_trace_v3": selected_uno_aggregate(uno_paths),
        "privacy": {
            "raw_states_published": False,
            "raw_html_published": False,
            "raw_urls_published": False,
            "raw_candidate_text_published": False,
            "raw_model_choices_published": False,
            "api_credentials_published": False,
            "identifiers": "dataset-scoped truncated SHA-256 hashes",
        },
    }
    write_json(args.output / "aggregate_metrics.json", aggregate)
    with (args.output / "per_example_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(args.output / "manifest.json", {
        "schema_version": "public-model-comparison-v1",
        "generated_on": str(date.today()),
        "files": ["aggregate_metrics.json", "per_example_metrics.jsonl", "report.md", "README.md"],
        "records": len(all_rows),
        "data_policy": "aggregate metrics and anonymized outcome metadata only; no browser states or model choices",
        "privacy": aggregate["privacy"],
    })


if __name__ == "__main__":
    main()
