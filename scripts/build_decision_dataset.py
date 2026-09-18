#!/usr/bin/env python3
"""Convert Browser Observation JSONL into a typed-decision training contract.

This is an adapter, not a claim that the legacy synthetic labels are suitable
for personal intent prediction. The generated metadata makes that distinction
explicit and keeps all prefixes before the labelled trajectory endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INTENTS = [
    "research_topic",
    "inspect_repository",
    "compare_products",
    "reply_to_email",
    "filter_and_export",
    "engage_with_post",
    "read_and_share_article",
    "dismiss_noise",
    "manage_settings",
    "unknown",
]
PREFIX_LENGTHS = (5, 10, 20, 50, 100, 200)


def family_split_map(family_ids: list[str]) -> dict[str, str]:
    """Assign whole source families to every split when the set is small.

    Hash-threshold splitting can accidentally produce an empty calibration or
    test split with only a handful of synthetic families. A stable sorted
    assignment makes that failure visible and guarantees coverage. The train
    split receives the remaining families after reserving one family for each
    non-train split.
    """
    ordered = sorted(set(family_ids), key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    if len(ordered) < 5:
        return {family: "train" for family in ordered}
    reserved = {
        family: split
        for family, split in zip(ordered[-4:], ("dev", "calibration", "test", "ood"))
    }
    return {family: reserved.get(family, "train") for family in ordered}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def family_for(row: dict[str, Any], intent: str) -> str:
    """Prefer a template family when v2 data provides one.

    Domain-only grouping is useful for the legacy smoke data but is too coarse
    for a generalization split: all search examples would otherwise be one
    family. Keeping the family ID explicit also makes leakage audits easier.
    """
    if row.get("template_id"):
        return f"v2:{row.get('template_id')}:{intent}"
    return f"legacy:{row.get('domain', 'unknown')}:{intent}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefixes", default=",".join(map(str, PREFIX_LENGTHS)))
    args = parser.parse_args()
    prefixes = tuple(sorted({int(value) for value in args.prefixes.split(",") if value.strip()}))
    if not prefixes or any(value <= 0 for value in prefixes):
        raise SystemExit("--prefixes must contain positive integers")

    output: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "dev", "calibration", "test", "ood")}
    rows = load_rows(args.input)
    family_ids = [family_for(row, row["trajectory_gold"].get("intent", "unknown")) for row in rows]
    split_by_family = family_split_map(family_ids)
    for row in rows:
        observation = row["observation"]
        events = row["events"]
        gold = row["trajectory_gold"]
        intent = gold.get("intent", "unknown")
        if intent not in INTENTS:
            intent = "unknown"
        family_id = family_for(row, intent)
        split = split_by_family[family_id]
        for requested_length in prefixes:
            # Do not repeat a short trajectory under multiple requested
            # lengths.  A repeated final state is both misleading for
            # calibration and rejected by NanoJev's unique-record check.
            if requested_length > len(events):
                continue
            prefix = events[:requested_length]
            if not prefix:
                continue
            state_id = f"legacy:{row['id']}:prefix:{requested_length}"
            decision_row = {
                "id": state_id,
                "state_id": state_id,
                "family_id": family_id,
                "split": split,
                "source": "uno-browser-bench-legacy-synthetic",
                "state": {
                    "observation": observation,
                    "events": prefix,
                    "feature_cutoff": {"event_count": len(prefix), "source_row_id": row["id"]},
                },
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": "What is the main task represented by the observed browser trajectory? Choose unknown if the evidence is insufficient.",
                        "criteria": {candidate: candidate.replace("_", " ") for candidate in INTENTS},
                    }
                },
                "gold": {"intent": intent},
                "gold_label_kind": {"intent": "observed_outcome"},
                "metadata": {
                    "source_group_id": f"legacy-row:{row['id']}",
                    "domain": row.get("domain", "unknown"),
                    "requested_prefix_length": requested_length,
                    "actual_prefix_length": len(prefix),
                    "trajectory_label_is_completed_task_label": True,
                    "personal_t0_prediction_ready": False,
                },
            }
            output[split].append(decision_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in output.items():
        target = args.output_dir / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for item in split_rows:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "browser-intent-decisions-v1",
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "records_by_split": {split: len(items) for split, items in output.items()},
        "intents": INTENTS,
        "prefixes": prefixes,
        "family_split": split_by_family,
        "warning": "This conversion supports pipeline validation and current-trajectory classification. It is not a personal T0 browser-intent dataset.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
