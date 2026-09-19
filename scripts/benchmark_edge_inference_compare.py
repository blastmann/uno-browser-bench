#!/usr/bin/env python3
"""Run a same-input Edge trace inference benchmark for NanoJev and Decider-2B.

Only the privacy-preserving Edge trace fields are read: action and coarse
page_type. Outputs contain trace ids and predictions, never URLs or titles.
The weak label comparison is intentionally named as such: next page category
is not semantic user intent ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


HORIZONS = (5, 10, 20, 50, 100, 200)
CHOICES = {
    "compare_products": "compare products",
    "dismiss_noise": "dismiss noise",
    "engage_with_post": "engage with post",
    "filter_and_export": "filter and export",
    "inspect_repository": "inspect repository",
    "manage_settings": "manage settings",
    "read_and_share_article": "read and share article",
    "reply_to_email": "reply to email",
    "research_topic": "research topic",
    "unknown": "unknown or insufficient evidence",
}
WEAK_TO_CHOICE = {
    "information_search": "research_topic",
    "development": "inspect_repository",
    "communication": "reply_to_email",
    "shopping": "compare_products",
    "social": "engage_with_post",
    "news": "read_and_share_article",
    "other": "unknown",
}


def read_edge_rows(path: Path, horizons: set[int], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("prefix_length", -1)) not in horizons:
                continue
            events = row.get("events") or []
            # Retain exactly the local extractor's non-sensitive coarse fields.
            compact_events = [
                {"step": int(event["step"]), "action": event["action"], "page_type": event["page_type"]}
                for event in events
            ]
            rows.append(
                {
                    "id": f"{row['trace_id']}:prefix:{row['prefix_length']}",
                    "prefix_length": int(row["prefix_length"]),
                    "state": {"events": compact_events},
                    "weak_gold": row["label"].get("weak_intent_proxy", "other"),
                    "page_type_gold": row["label"].get("page_type", "other"),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def question() -> dict[str, Any]:
    return {
        "intent": {
            "type": "choice",
            "instructions": "What is the main task represented by this coarse browser trace? Choose unknown if the evidence is insufficient.",
            "criteria": CHOICES,
        }
    }


def make_request(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": row["id"], "state": row["state"], "questions": question()}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_stats(torch: Any) -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
    }


def extract_answer(answers: dict[str, Any]) -> tuple[str | None, float | None]:
    answer = answers.get("intent") or {}
    choice = answer.get("choice")
    if choice is None:
        choice = answer.get("value")
    confidence = answer.get("confidence")
    if confidence is None and isinstance(answer.get("probabilities"), dict) and choice is not None:
        confidence = answer["probabilities"].get(choice)
    return choice, float(confidence) if confidence is not None else None


def load_engine(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch

    if args.model == "nanojev":
        nanojev_root = args.nanojev_root.resolve()
        sys.path.insert(0, str(nanojev_root / "scripts"))
        from predict_toy_decisions import DecisionPredictor

        engine = DecisionPredictor(
            args.nanojev_checkpoint,
            max_length=args.max_length,
            precision="bf16",
            disable_native_triton=args.disable_native_triton,
        )
        return engine, torch

    decider_root = args.decider_root.resolve()
    sys.path.insert(0, str(decider_root))
    from decider.infer import Decider

    engine = Decider(str(args.decider_model), device="cuda", use_graphs=False)
    return engine, torch


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_edge_rows(args.input, set(args.horizons), args.limit)
    if not rows:
        raise SystemExit("没有找到指定 horizon 的 Edge trace")
    engine, torch = load_engine(args)
    requests = [make_request(row) for row in rows]

    warmup_count = min(args.warmup, len(requests))
    for request in requests[:warmup_count]:
        if args.model == "nanojev":
            engine.predict({"states": [request]}, batch_questions=1)
        else:
            engine.system_one(request["state"], request["questions"], max_state_tokens=args.max_state_tokens)
    sync_cuda(torch)
    torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row, request in zip(rows, requests):
        sync_cuda(torch)
        one_started = time.perf_counter()
        if args.model == "nanojev":
            response = engine.predict({"states": [request]}, batch_questions=1)
            answers = response["states"][0]["answers"]
        else:
            response = engine.system_one(request["state"], request["questions"], max_state_tokens=args.max_state_tokens)
            answers = response["answers"]
        sync_cuda(torch)
        elapsed = time.perf_counter() - one_started
        latencies.append(elapsed)
        choice, confidence = extract_answer(answers)
        predictions.append(
            {
                "id": row["id"],
                "prefix_length": row["prefix_length"],
                "page_type_gold": row["page_type_gold"],
                "weak_gold": row["weak_gold"],
                "weak_gold_mapped_choice": WEAK_TO_CHOICE.get(row["weak_gold"], "unknown"),
                "choice": choice,
                "confidence": confidence,
                "latency_ms": round(elapsed * 1000, 3),
            }
        )
    total = time.perf_counter() - started

    by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in args.horizons:
        selected = [p for p in predictions if p["prefix_length"] == horizon]
        values = [p["latency_ms"] for p in selected]
        correct = [p for p in selected if p["choice"] == p["weak_gold_mapped_choice"]]
        by_horizon[str(horizon)] = {
            "n": len(selected),
            "weak_proxy_top1": round(len(correct) / len(selected), 6) if selected else None,
            "mean_ms": round(statistics.mean(values), 3) if values else None,
            "p50_ms": round(percentile(values, 0.50), 3) if values else None,
            "p95_ms": round(percentile(values, 0.95), 3) if values else None,
        }

    all_correct = [p for p in predictions if p["choice"] == p["weak_gold_mapped_choice"]]
    result = {
        "schema_version": "edge-inference-compare-v1",
        "model": args.model,
        "input": str(args.input),
        "rows": len(rows),
        "horizons": args.horizons,
        "warmup_rows": warmup_count,
        "precision": "bf16",
        "model_load_excluded": True,
        "timing_scope": "one state inference plus host preprocessing and CUDA synchronization; no network",
        "elapsed_s": round(total, 6),
        "throughput_rows_per_s": round(len(rows) / total, 6) if total else None,
        "latency_ms": {
            "mean": round(statistics.mean(latencies) * 1000, 3),
            "p50": round(percentile(latencies, 0.50) * 1000, 3),
            "p95": round(percentile(latencies, 0.95) * 1000, 3),
            "max": round(max(latencies) * 1000, 3),
        },
        "weak_proxy": {
            "label": "observed next page category mapped to a coarse choice; not semantic intent ground truth",
            "top1": round(len(all_correct) / len(predictions), 6) if predictions else None,
        },
        "by_horizon": by_horizon,
        "memory": memory_stats(torch),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.predictions.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("nanojev", "decider"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--nanojev-root", type=Path, required=True)
    parser.add_argument("--nanojev-checkpoint", type=Path, required=True)
    parser.add_argument("--decider-root", type=Path, required=True)
    parser.add_argument("--decider-model", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-state-tokens", type=int, default=4096)
    parser.add_argument("--disable-native-triton", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
