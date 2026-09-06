# uno-browser-bench

在 Windows 11 + WSL2 Ubuntu 上对 IFM Uno 与 K2-Horizon-0.9B 做浏览器语义层实测。

## 目标

- 记录 Windows/WSL2/GPU/CUDA/Python/显存环境。
- 对普通 AR 与 Uno Linear sampler 做 256/1K/4K/8K 输入、128 输出的 A/B。
- 用不含原始 HTML 的 Browser Observation 做实体抽取和行为序列 benchmark。
- 记录 JSON valid、schema valid、实体 P/R/F1、核心字段 exact match、intent accuracy、step F1，以及长轨迹退化。

## 官方路径

Uno 官方当前 README（仓库 commit 见 `results/environment.json`）要求 Python 3.10、PyTorch 2.11.0 + CUDA 12.8，并提供 Linux x86_64 的 FlashAttention-2 wheel；FA2 足够跑 Linear sampler，Tree sampler 才额外需要 FA3。本实验第一阶段固定 `attention-backend=fa2`，不引入 Tree sampler。

模型：

- AR 基线：`IFM/K2-Horizon-0.9B`
- Uno：`IFM/K2-Horizon-0.9B` + `IFM/K2-Horizon-0.9B-Uno`

## 目录

```text
uno-browser-bench/
├─ data/                 # 可重复生成的 JSONL 数据集
├─ environment/          # 环境快照
├─ logs/                 # 安装与运行日志
├─ results/              # CSV/JSON/报告
├─ scripts/              # 生成、推理、评测、汇总脚本
└─ vendor/uno/           # Uno 官方源码快照
```

## 运行方式（WSL2 Ubuntu）

```bash
python scripts/generate_dataset.py --output data/browser_semantics.jsonl --count 200 --seed 20260906
python scripts/measure_environment.py --output results/environment.json
python scripts/run_model_benchmark.py --mode ar --dataset data/browser_semantics.jsonl --output results/ar_semantics.jsonl
python scripts/run_model_benchmark.py --mode uno --dataset data/browser_semantics.jsonl --output results/uno_semantics.jsonl
python scripts/evaluate_outputs.py --input results/ar_semantics.jsonl --output results/ar_quality.json
python scripts/evaluate_outputs.py --input results/uno_semantics.jsonl --output results/uno_quality.json
```

性能 A/B 由 `scripts/run_speed_bench.py` 调用两个 runner，输出：

- `results/speed_*_wsl_tokens.jsonl`
- `results/speed_*_native_*_tokens.jsonl`
- `results/speed_summary*.csv`
- `results/report.md`

默认正式运行是 warmup 5 + measured 30；如果显存或时间不足，脚本参数会把缩减写入结果元数据，不能静默改变样本数。

speed harness 会记录 tokenizer 实际输入长度。当前四档目标 256/1K/4K/8K 实际为 552/1432/4941/9627 tokens，确保每档不低于目标值。

## 原生 Windows 对照

原生 Windows AR 与 Uno 都使用 benchmark 目录内的 `.venv-win-ar`，不修改系统 Python 或 WSL 环境：

```powershell
pwsh -File scripts/run_native_windows_ar.ps1
pwsh -File scripts/run_native_windows_uno.ps1
```

结果写入 `results/speed_ar_native_fa2_aligned_tokens.jsonl` 与 `results/speed_uno_native_fixed_tokens.jsonl`。原生 Windows 已完成 4 档、warmup 5 + formal 30；环境为 CUDA Toolkit 12.8.1、PyTorch 2.11.0+cu128、源码构建的 FlashAttention 2.8.3、`triton-windows`，Uno 单卡使用 Gloo process group 兼容补丁。完整证据见 `results/native_uno_compatibility.json`。

原生 Windows 的正式语义质量命令：

```powershell
$env:NANO_VLLM_DIST_BACKEND = "gloo"
python scripts/run_model_benchmark.py --mode uno --dataset data/browser_semantics.jsonl --output results/uno_semantics_native_fixed_50.jsonl --task all --limit 50 --batch-size 1 --max-new-tokens 256 --gpu-memory-utilization 0.90
python scripts/evaluate_outputs.py --input results/uno_semantics_native_fixed_50.jsonl --output results/uno_quality_native_fixed_50.json
```

## 结果解释

`TTFT` 以模型首次产出 token 前的计时为准；`TPS` 是生成 token / decode elapsed；`TPF` 对 Uno 使用官方 decoder `accepts / forwards`，AR 固定为 1.0（每次 forward 产生一个新 token）。峰值显存来自 PyTorch allocator，GPU utilization 是采样窗口内的 nvidia-smi 观测值。

对齐后的实测结果：原生 Windows Uno/AR 平均约 4.56x，WSL2 约 4.69x；原生/WSL TPS 比例 AR 为 1.03–1.06、Uno 为 1.01–1.02。这个结果只代表同一测试主机上的观测，不是所有 Windows/WSL2 硬件的通用结论。

公开发布版本会省略本机主机名、用户名、绝对路径、详细驱动/磁盘信息、模型缓存、虚拟环境和运行日志。

## 已知边界

- 本 benchmark 只测文本结构化 observation，不测截图视觉能力。
- 语义数据集是确定性合成/半真实模板，不等同于真实用户流量；报告会把它标为 smoke/合成结果。
- 0.9B 模型的 JSON 能力强依赖提示词与输出截断；所有无效 JSON 和 schema 错误都会保留原始输出以便复盘。
