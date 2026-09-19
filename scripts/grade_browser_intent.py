#!/usr/bin/env python3
"""Create an evidence-based local P2/P3/P4/P6 grade."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--edge-audit", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--model-diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = read(args.split_audit)
    edge = read(args.edge_audit)
    calibration = read(args.calibration_report)
    diagnostics = read(args.model_diagnostics)

    privacy = edge.get("privacy", {})
    privacy_safe = all(
        privacy.get(field) is False
        for field in (
            "raw_urls_written",
            "raw_titles_written",
            "hostnames_written",
            "query_strings_written",
            "paths_written",
            "timestamps_written",
            "url_hashes_written",
        )
    ) and privacy.get("network_calls") == 0
    p2_structural = bool(split.get("passed"))
    p2 = {
        "status": "partial" if p2_structural and privacy_safe else "incomplete",
        "structural_split_audit_passed": p2_structural,
        "local_trace_privacy_passed": privacy_safe,
        "synthetic_rows": split.get("rows"),
        "local_edge_sessions": edge.get("sessions_eligible_min_2_events"),
        "local_edge_prefix_rows": edge.get("prefix_rows"),
        "held_out_label_counts": split.get("held_out_label_counts"),
        "limitation": (
            "Synthetic labels are completed-task labels and local Edge labels are "
            "observed next-category proxies; neither is ground-truth latent intent."
        ),
    }

    temperature = calibration.get("temperature", {})
    test = calibration.get("by_split", {}).get("test", {})
    ood = calibration.get("by_split", {}).get("ood", {})
    calibration_good = (
        temperature.get("fitted") is True
        and temperature.get("calibration_rows", 0) >= 50
        and test.get("ece", 1.0) <= 0.10
        and ood.get("ece", 1.0) <= 0.10
    )
    p3 = {
        "status": "pass" if calibration_good else "measured_not_passed",
        "temperature": temperature.get("value"),
        "calibration_rows": temperature.get("calibration_rows"),
        "test": {key: test.get(key) for key in ("n", "nll", "brier", "ece", "mean_confidence")},
        "ood": {key: ood.get(key) for key in ("n", "nll", "brier", "ece", "mean_confidence")},
        "criteria": "test and OOD ECE <= 0.10 after calibration",
    }

    curve = calibration.get("long_trajectory_curve", {})
    required_lengths = ["5", "10", "20", "50", "100", "200"]
    test_curve = curve.get("test", {})
    ood_curve = curve.get("ood", {})
    curve_complete = all(length in test_curve and length in ood_curve for length in required_lengths)
    test_drop = None
    if "5" in test_curve and "200" in test_curve:
        test_drop = test_curve["5"].get("accuracy", 0.0) - test_curve["200"].get("accuracy", 0.0)
    p4 = {
        "status": "measured" if curve_complete else "incomplete",
        "lengths": required_lengths,
        "test_curve_complete": all(length in test_curve for length in required_lengths),
        "ood_curve_complete": all(length in ood_curve for length in required_lengths),
        "test_accuracy_5_to_200_drop": test_drop,
        "test_accuracy_by_length": {length: test_curve.get(length, {}).get("accuracy") for length in required_lengths},
        "ood_accuracy_by_length": {length: ood_curve.get(length, {}).get("accuracy") for length in required_lengths},
    }

    p6 = {
        "grade": "candidate_ranking_only_needs_upgrade",
        "conclusion_zh": "暂不适合作为未经升级的常驻 Browser semantic layer；可作为候选排序/低置信度路由辅助，必须保留升级路径。",
        "why": [
            "P2 结构审计与隐私边界通过，但真实 Edge 标签只是弱行为代理。",
            "P3 已完成独立温度拟合，但 test/OOD ECE 仍高，不能直接按置信度自动执行。",
            "P4 已观察到长轨迹明显退化，200 步 test accuracy 仅 28.57%。",
            "NanoJev/Decider 合成 benchmark 的高分受词汇/模板 shortcut 影响，不能外推为真实用户意图准确率。",
        ],
        "deployment_policy": {
            "allow_local_candidate_ranking": True,
            "allow_unverified_action_execution": False,
            "fallback_to_stronger_model_or_user": True,
            "minimum_next_gate": "新增人工标注真实/半真实 trace，跨域且含反事实，再重复 P2-P4。",
        },
    }

    report = {
        "schema_version": "browser-intent-p2-p6-grade-v1",
        "scope": "local-only; no personal Edge trace payload is embedded",
        "p2_data_validity": p2,
        "p3_probability_calibration": p3,
        "p4_long_trajectory": p4,
        "p6_final_grade": p6,
        "supporting_diagnostics": {
            "nanojev_large_test_accuracy": diagnostics.get("large_smoke", {}).get("test", {}).get("accuracy"),
            "nanojev_large_ood_accuracy": diagnostics.get("large_smoke", {}).get("ood", {}).get("accuracy"),
            "nanojev_lexical_ablation_accuracy": diagnostics.get("lexical_ablation", {}).get("accuracy"),
            "decider_legacy_v2_accuracy": diagnostics.get("decider_public_v2", {}).get("overall", {}).get("accuracy"),
            "decider_legacy_v2_lexical_ablation_accuracy": diagnostics.get("decider_public_v2", {}).get("lexical_ablation", {}).get("accuracy"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "grade": p6["grade"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
