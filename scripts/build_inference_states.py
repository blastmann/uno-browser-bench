#!/usr/bin/env python3
"""Build NanoJev predictor input from the decision JSONL contract."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--limit-per-length", type=int, default=0)
    args = parser.parse_args()
    selected: list[dict] = []
    by_length: dict[str, list[dict]] = defaultdict(list)
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        path = args.input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            length = str(row["state_id"].rsplit(":", 1)[-1])
            by_length[length].append({key: row[key] for key in ("id", "state", "questions")})
    for length in sorted(by_length, key=lambda value: int(value) if value.isdigit() else value):
        rows = by_length[length]
        if args.limit_per_length:
            rows = rows[: args.limit_per_length]
        selected.extend(rows)
    payload = {"states": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "states": len(selected), "lengths": {key: len(value) for key, value in sorted(by_length.items())}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
