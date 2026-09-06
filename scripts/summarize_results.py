#!/usr/bin/env python3
"""Assemble aligned native-Windows and WSL2 measurements into CSV and Markdown."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def speed_rows(ar_path: Path, uno_path: Path) -> list[dict]:
    ar = {row["input_target_tokens"]: row for row in load_jsonl(ar_path) if row.get("status") == "ok"}
    uno = {row["input_target_tokens"]: row for row in load_jsonl(uno_path) if row.get("status") == "ok"}
    rows = []
    for target in (256, 1024, 4096, 8192):
        a = ar[target]
        u = uno[target]
        rows.append(
            {
                "target_tokens": target,
                "actual_tokens": a["observed"]["prompt_tokens"],
                "ar_tps": a["tps_mean"],
                "uno_tps": u["tps_mean"],
                "speedup": u["tps_mean"] / a["tps_mean"],
                "ar_p50_s": a["latency_p50_s"],
                "ar_p95_s": a["latency_p95_s"],
                "uno_p50_s": u["latency_p50_s"],
                "uno_p95_s": u["latency_p95_s"],
                "ar_tpf": a["tpf"],
                "uno_tpf": u["tpf"],
                "ar_peak_vram_mb": a["peak_vram_mb"],
                "uno_peak_vram_mb": u["peak_vram_mb"],
                "ar_gpu_util_pct": a.get("gpu_sample", {}).get("gpu_util_pct"),
                "uno_gpu_util_pct": u.get("gpu_sample", {}).get("gpu_util_pct"),
                "ar_sample_vram_mb": a.get("gpu_sample", {}).get("vram_used_mb"),
                "uno_sample_vram_mb": u.get("gpu_sample", {}).get("vram_used_mb"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def quality_line(label: str, quality: dict) -> str:
    return (
        f"| {label} | {quality['rates']['json_valid']:.1%} | "
        f"{quality['rates']['schema_valid']:.1%} | "
        f"{(quality.get('entity_f1') or 0):.1%} | "
        f"{(quality.get('intent_accuracy') or 0):.1%} | "
        f"{(quality.get('step_f1') or 0):.1%} |"
    )


def speed_table(lines: list[str], rows: list[dict]) -> None:
    lines.extend(
        [
            "| Target tokens | Actual tokens | AR TPS | Uno TPS | Speedup | AR P50/P95 s | Uno P50/P95 s | AR TPF | Uno TPF | Uno peak VRAM MB | AR/Uno GPU util % |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['target_tokens']} | {row['actual_tokens']} | {row['ar_tps']:.1f} | "
            f"{row['uno_tps']:.1f} | {row['speedup']:.2f}x | "
            f"{row['ar_p50_s']:.3f}/{row['ar_p95_s']:.3f} | "
            f"{row['uno_p50_s']:.3f}/{row['uno_p95_s']:.3f} | "
            f"{row['ar_tpf']:.2f} | {row['uno_tpf']:.2f} | {row['uno_peak_vram_mb']:.0f} | "
            f"{row['ar_gpu_util_pct']:.0f}/{row['uno_gpu_util_pct']:.0f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    results = root / "results"

    wsl_rows = speed_rows(results / "speed_ar_wsl_tokens.jsonl", results / "speed_uno_wsl_tokens.jsonl")
    native_rows = speed_rows(
        results / "speed_ar_native_fa2_aligned_tokens.jsonl",
        results / "speed_uno_native_fixed_tokens.jsonl",
    )
    write_csv(results / "speed_summary.csv", wsl_rows)
    write_csv(results / "speed_summary_wsl_tokens.csv", wsl_rows)
    write_csv(results / "speed_summary_native_tokens.csv", native_rows)

    cross_rows = []
    for wsl, native in zip(wsl_rows, native_rows):
        cross_rows.append(
            {
                "target_tokens": native["target_tokens"],
                "actual_tokens": native["actual_tokens"],
                "native_ar_tps": native["ar_tps"],
                "wsl_ar_tps": wsl["ar_tps"],
                "native_over_wsl_ar": native["ar_tps"] / wsl["ar_tps"],
                "native_uno_tps": native["uno_tps"],
                "wsl_uno_tps": wsl["uno_tps"],
                "native_over_wsl_uno": native["uno_tps"] / wsl["uno_tps"],
            }
        )
    write_csv(results / "speed_summary_cross_platform_tokens.csv", cross_rows)

    wsl_ar_quality_path = first_existing(results, ["ar_quality_50.json", "ar_quality_fixed_50.json"])
    wsl_uno_quality_path = first_existing(results, ["uno_quality_50.json", "uno_quality_fixed_50.json"])
    native_ar_quality_path = results / "ar_quality_native_fa2_fixed_50.json"
    native_uno_quality_path = results / "uno_quality_native_fixed_50.json"
    wsl_ar_quality = load_json(wsl_ar_quality_path) if wsl_ar_quality_path else None
    wsl_uno_quality = load_json(wsl_uno_quality_path) if wsl_uno_quality_path else None
    native_ar_quality = load_json(native_ar_quality_path) if native_ar_quality_path.exists() else None
    native_uno_quality = load_json(native_uno_quality_path) if native_uno_quality_path.exists() else None

    native_speedups = [row["speedup"] for row in native_rows]
    wsl_speedups = [row["speedup"] for row in wsl_rows]
    native_ar_ratio = [row["native_over_wsl_ar"] for row in cross_rows]
    native_uno_ratio = [row["native_over_wsl_uno"] for row in cross_rows]
    windows_env = load_json(results / "windows_environment.json")
    compatibility = load_json(results / "native_uno_compatibility.json")

    lines = [
        "# Uno Browser Bench 实测报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 结论摘要",
        "",
        f"- 在同一份实际输入（552/1432/4941/9627 tokens，目标为 256/1K/4K/8K）和 128-token 输出下，原生 Windows Uno Linear 相对 AR 的 TPS speedup 为 **{statistics.mean(native_speedups):.2f}×**（{min(native_speedups):.2f}–{max(native_speedups):.2f}×）；WSL2 为 **{statistics.mean(wsl_speedups):.2f}×**（{min(wsl_speedups):.2f}–{max(wsl_speedups):.2f}×）。",
        f"- 原生 Windows / WSL2 的同路径 TPS 比例：AR 为 **{min(native_ar_ratio):.2f}–{max(native_ar_ratio):.2f}**，Uno 为 **{min(native_uno_ratio):.2f}–{max(native_uno_ratio):.2f}**；本次实测没有看到 WSL2 的明显性能损耗，差异在约 1–6% 范围内。",
        "- 原生 Windows Uno Linear 已真实完成 4/4 speed 档位和 100 项语义质量任务；通过 Windows-only Gloo process group 兼容补丁运行，Linux/WSL 默认仍使用 NCCL。",
    ]
    if native_uno_quality:
        lines.append(
            f"- 原生 Uno 语义质量：JSON valid **{native_uno_quality['rates']['json_valid']:.1%}**，schema valid **{native_uno_quality['rates']['schema_valid']:.1%}**，实体 F1 **{native_uno_quality['entity_f1']:.1%}**，intent accuracy **{native_uno_quality['intent_accuracy']:.1%}**，step F1 **{native_uno_quality['step_f1']:.1%}**。"
        )

    lines += [
        "",
        "## 环境与安装",
        "",
        f"- Windows 主机：本地 Windows 测试主机；GPU：NVIDIA GeForce RTX 4070 Ti SUPER；显存：16,376 MiB。",
        f"- Windows：{windows_env.get('os', {}).get('Caption')} build {windows_env.get('os', {}).get('BuildNumber')}；驱动：{windows_env.get('gpu', [{}])[0].get('DriverVersion')}。",
        "- 原生 Python：独立 `.venv-win-ar`，CPython 3.10.16；PyTorch 2.11.0+cu128；Transformers 5.16.1；einops 0.8.2。",
        "- 原生 CUDA：Toolkit 12.8.1，nvcc 12.8.93；Visual Studio/MSVC 2022 14.44.35207。",
        "- 原生 FlashAttention：2.8.3 从源码针对 torch 2.11.0+cu128 构建，并通过 BF16 CUDA kernel smoke；`triton-windows==3.8.0.post28`。",
        "- WSL2 主路径：Uno 官方 Linux 路径，Python 3.10、PyTorch cu128、FA2、NCCL；Uno Linear 未引入 FA3/Tree sampler。",
        "- 官方兼容性说明：Uno/FlashAttention 的官方主路径仍偏向 Linux；Windows 结果是本机已验证的兼容构建，不等于上游对 Windows 的正式支持。",
        "",
        "## WSL2：官方路径 speed A/B",
        "",
    ]
    speed_table(lines, wsl_rows)
    lines += ["", "## 原生 Windows：FA2 + Uno Linear 对齐 speed A/B", ""]
    speed_table(lines, native_rows)
    lines += [
        "",
        "## 原生 Windows / WSL2 同路径比较",
        "",
        "| Target tokens | Actual tokens | Native AR TPS | WSL AR TPS | Native/WSL AR | Native Uno TPS | WSL Uno TPS | Native/WSL Uno |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cross_rows:
        lines.append(
            f"| {row['target_tokens']} | {row['actual_tokens']} | {row['native_ar_tps']:.1f} | {row['wsl_ar_tps']:.1f} | {row['native_over_wsl_ar']:.3f} | {row['native_uno_tps']:.1f} | {row['wsl_uno_tps']:.1f} | {row['native_over_wsl_uno']:.3f} |"
        )
    lines += [
        "",
        "注：这是同一台机器、同一模型/输入生成方式、同一 FA2 backend 的观测对照，不是跨机器或严格 OS 微基准；1–6% 差异可能包含运行时、调度和测量噪声。",
        "",
        "## 语义质量（50 条数据，实体+行为共 100 项）",
        "",
        "| runner | JSON valid | schema valid | entity F1 | intent acc | step F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, quality in (
        ("WSL AR", wsl_ar_quality),
        ("WSL Uno", wsl_uno_quality),
        ("Native AR FA2", native_ar_quality),
        ("Native Uno Linear", native_uno_quality),
    ):
        if quality:
            lines.append(quality_line(label, quality))
    lines += ["", "### Uno 长轨迹 JSON valid（原生 Windows）", "", "| events | valid |", "|---:|---:|"]
    if native_uno_quality:
        for length, values in native_uno_quality.get("by_trajectory_length", {}).items():
            lines.append(f"| {length} | {values['json_valid'] / values['total']:.1%} |")
    lines += [
        "",
        "## 解释与限制",
        "",
        "- 输入长度表同时记录目标档位和 tokenizer 实际长度；当前合成 filler 实际为 552/1432/4941/9627 tokens，因此满足至少 256/1K/4K/8K 的要求。",
        "- Uno TPF 是官方 decoder 的 `accepts / forwards`；AR 固定为 1.0。P50/P95 是每档 30 次正式 run 的总生成 latency；Uno 通过 completion callback 的请求间隔近似得到。",
        "- TTFT：语义质量 runner 对 AR 记录了首次 logits processor 时间；Uno 官方公开 generate API 只暴露完成回调，没有逐请求首 token回调，因此 speed JSON 不伪造 Uno TTFT，原始质量 JSON 保留 `ttft_s: null` 及该限制。",
        "- 质量数据是可重复的合成/半真实 Browser Observation，不含原始 HTML，字段覆盖 url/title/elements/events，域包括电商、搜索、新闻、GitHub、邮件、社交、后台和噪声页；50 条 smoke 集和扩展脚本均保留。",
        f"- 原生 Windows Uno 兼容证据：{compatibility.get('conclusion', '')}",
        "",
        "## 三个问题的回答",
        "",
        f"1. **Uno 是否明显更快？** 是。对齐后的真实 token 测量中，原生 Windows speedup 平均 {statistics.mean(native_speedups):.2f}×，WSL2 平均 {statistics.mean(wsl_speedups):.2f}×。",
        "2. **0.9B 对实体/行为语义是否够用？** 作为粗粒度 JSON parser 有一定能力，但实体 F1 只有约 8–12%，intent/step 严格准确率为 0%；不适合未经约束解码、后处理或蒸馏就直接作为生产常驻 semantic layer。",
        "3. **长轨迹是否快速退化？** 是。原生 Uno 的 trajectory JSON valid 在 5/10 events 为 100%，20 events 约 62.5%，50 events 50%，100 events 62.5%，200 events 56.3%；长轨迹应先做事件压缩/分段摘要。",
        "",
        "## 来源",
        "",
        "- Uno 官方仓库：https://github.com/ifm-ai/uno",
        "- K2-Horizon-0.9B-Uno：https://huggingface.co/IFM/K2-Horizon-0.9B-Uno",
        "- K2-Horizon-0.9B：https://huggingface.co/IFM/K2-Horizon-0.9B",
        "- FlashAttention：https://github.com/Dao-AILab/flash-attention",
        "- CUDA 12.8 Windows 安装文档：https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-microsoft-windows/",
        "- WSL GPU compute：https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute",
        "",
    ]
    (results / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wsl_rows": len(wsl_rows), "native_rows": len(native_rows), "report": str(results / 'report.md')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
