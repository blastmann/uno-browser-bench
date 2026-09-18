#!/usr/bin/env python3
"""Create a structure/action-only ablation for leakage diagnosis."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def ablate(row: dict) -> dict:
    row = copy.deepcopy(row)
    state = row["state"]
    observation = state["observation"]
    observation["url"] = "https://example.test/"
    observation["title"] = "<page>"
    observation["elements"] = [{"id": "<element>", "role": element.get("role", ""), "text": "<element>"} for element in observation.get("elements", [])]
    observation["events"] = [{"t": event.get("t", index), "action": event.get("action", ""), "target": "<target>"} for index, event in enumerate(observation.get("events", []))]
    state["events"] = [{"t": event.get("t", index), "action": "<action>", "target": "<target>"} for index, event in enumerate(state.get("events", []))]
    return {key: row[key] for key in ("id", "state", "questions")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="test,ood")
    args = parser.parse_args()
    states = []
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        path = args.input_dir / f"{split}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    states.append(ablate(json.loads(line)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"states": states}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "states": len(states)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
