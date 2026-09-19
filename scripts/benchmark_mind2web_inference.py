#!/usr/bin/env python3
"""Compare NanoJev and Decider-2B on the same prepared Mind2Web test rows."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("test_task", "test_website", "test_domain")


def read_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = input_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def request(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": row["id"], "state": row["state"], "questions": row["questions"]}


def sync_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def memory_stats(torch: Any) -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
    }


def load_engine(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch

    if args.model == "nanojev":
        sys.path.insert(0, str(args.nanojev_root.resolve() / "scripts"))
        from predict_toy_decisions import DecisionPredictor

        return DecisionPredictor(
            args.nanojev_checkpoint,
            max_length=args.max_length,
            precision="bf16",
            disable_native_triton=args.disable_native_triton,
        ), torch

    sys.path.insert(0, str(args.decider_root.resolve()))
    from decider.infer import Decider

    return Decider(str(args.decider_model), device="cuda", use_graphs=False), torch


def answer_parts(answer: dict[str, Any]) -> tuple[str | None, list[str]]:
    choice = answer.get("choice") or answer.get("value")
    probabilities = answer.get("probabilities") or {}
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    return choice, ranked


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        by_operation[row["operation"]].append(row)

    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"rows": 0}
        task_values: defaultdict[str, list[int]] = defaultdict(list)
        for row in group:
            task_values[row["task_id"]].append(int(row["correct"]))
        macro_step = statistics.mean(statistics.mean(v) for v in task_values.values())
        sampled_success = sum(all(v) for v in task_values.values()) / len(task_values)
        latencies = [row["latency_ms"] for row in group]
        return {
            "rows": len(group),
            "tasks": len(task_values),
            "top1_accuracy": round(sum(row["correct"] for row in group) / len(group), 6),
            "top3_accuracy": round(sum(row["top3"] for row in group) / len(group), 6),
            "macro_step_accuracy": round(macro_step, 6),
            "sampled_task_all_steps_accuracy": round(sampled_success, 6),
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 3),
                "p50": round(percentile(latencies, 0.50), 3),
                "p95": round(percentile(latencies, 0.95), 3),
                "max": round(max(latencies), 3),
            },
        }

    return {
        "overall": one(rows),
        "by_split": {key: one(value) for key, value in by_split.items()},
        "by_operation": {key: one(value) for key, value in by_operation.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(args.input_dir)
    if args.limit:
        rows = rows[: args.limit]
    engine, torch = load_engine(args)
    requests = [request(row) for row in rows]
    for item in requests[: min(args.warmup, len(requests))]:
        if args.model == "nanojev":
            engine.predict({"states": [item]}, batch_questions=1)
        else:
            engine.system_one(item["state"], item["questions"], max_state_tokens=args.max_state_tokens)
    sync_cuda(torch)
    torch.cuda.reset_peak_memory_stats()

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row, item in zip(rows, requests):
        sync_cuda(torch)
        one_started = time.perf_counter()
        if args.model == "nanojev":
            response = engine.predict({"states": [item]}, batch_questions=1)
            answers = response["states"][0]["answers"]
        else:
            response = engine.system_one(item["state"], item["questions"], max_state_tokens=args.max_state_tokens)
            answers = response["answers"]
        sync_cuda(torch)
        latency_ms = (time.perf_counter() - one_started) * 1000
        choice, ranked = answer_parts(answers["target"])
        predictions.append(
            {
                "id": row["id"],
                "split": row["split"],
                "task_id": row["task_id"],
                "operation": row["operation"],
                "candidate_count": row["candidate_count"],
                "gold": row["gold"],
                "choice": choice,
                "correct": int(choice == row["gold"]),
                "top3": int(row["gold"] in ranked[:3]),
                "latency_ms": round(latency_ms, 3),
            }
        )
    elapsed = time.perf_counter() - started
    summary = summarize(predictions)
    result = {
        "schema_version": "mind2web-inference-comparison-v1",
        "model": args.model,
        "input_dir": str(args.input_dir),
        "rows": len(rows),
        "warmup_rows": min(args.warmup, len(rows)),
        "precision": "bf16",
        "model_load_excluded": True,
        "timing_scope": "one state inference plus host preprocessing and CUDA synchronization; no network",
        "context_limit": args.max_length,
        "elapsed_s": round(elapsed, 6),
        "throughput_rows_per_s": round(len(rows) / elapsed, 6) if elapsed else None,
        "memory": memory_stats(torch),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "metrics": summary,
        "note": "This is Mind2Web candidate element selection, not full browser task execution. Raw HTML is not sent to either model.",
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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--nanojev-root", type=Path, required=True)
    parser.add_argument("--nanojev-checkpoint", type=Path, required=True)
    parser.add_argument("--decider-root", type=Path, required=True)
    parser.add_argument("--decider-model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-state-tokens", type=int, default=8192)
    parser.add_argument("--disable-native-triton", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
