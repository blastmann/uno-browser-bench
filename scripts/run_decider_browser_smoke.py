#!/usr/bin/env python3
"""Run the released decider-2b on the public Browser Observation decision set.

This is intentionally an inference-only adapter. Model weights and caches stay
outside the repository; the output contains public synthetic states only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_rows(input_dir: Path, splits: list[str], limit: int) -> list[dict]:
    rows: list[dict] = []
    for split in splits:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def load_states(path: Path, limit: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for state in payload.get("states", []):
        rows.append({"id": state["id"], "state": state["state"], "questions": state["questions"]})
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-state-tokens", type=int, default=2048)
    args = parser.parse_args()
    if not args.input_json and not args.input_dir:
        parser.error("one of --input-dir or --input-json is required")

    from decider.infer import Decider

    rows = load_states(args.input_json, args.limit) if args.input_json else load_rows(args.input_dir, [s.strip() for s in args.splits.split(",") if s.strip()], args.limit)
    print(json.dumps({"rows": len(rows), "model_path": str(args.model_path)}, ensure_ascii=False), flush=True)
    model = Decider(str(args.model_path), device="cuda", use_graphs=False)
    states: list[dict] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        answers = model.system_one(row["state"], row["questions"], max_state_tokens=args.max_state_tokens)
        states.append({"id": row["id"], "answers": answers["answers"], "usage": answers.get("usage", {})})
        if index == 1 or index % 10 == 0:
            print(json.dumps({"completed": index, "elapsed_s": round(time.perf_counter() - started, 2)}, ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "browser-intent-decider-inference-v1",
        "model": "Mapika/decider-2b",
        "rows": len(states),
        "elapsed_s": elapsed,
        "states": states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(states), "elapsed_s": round(elapsed, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
