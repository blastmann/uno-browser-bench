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

## Browser Intent / Jev 验证进度

本仓库现在同时包含 Uno 速度 benchmark 和 Browser Intent / typed-decision 实验。Browser Observation 只使用 `url/title/elements/events`，不把原始 HTML 送进模型。

```powershell
# 生成并审计 v2 数据集（相对路径输出，不含个人浏览记录）
python scripts/generate_intent_dataset_v2.py --output data/browser_intent/observation_v2.jsonl --count-per-scenario 10 --seed 20260918
python scripts/audit_browser_dataset.py --input data/browser_intent/observation_v2.jsonl --output data/browser_intent/audit_v2.json
python scripts/build_decision_dataset.py --input data/browser_intent/observation_v2.jsonl --output-dir data/browser_intent/decisions_v2_public --prefixes 5,10,20,50,100,200
python scripts/validate_decision_contract.py --input-dir data/browser_intent/decisions_v2_public --output data/browser_intent/contract_v2.json
```

NanoJev 的真实 Windows GPU smoke 需要仓库外的本地 `.venv-win-ar` 和 `.tools/NanoJev`；公开的可复核摘要见 [`results/browser_intent_smoke_summary.json`](results/browser_intent_smoke_summary.json)。该摘要明确区分了 smoke 结果、未完成的 decider 权重推理和不能外推的指标。

```powershell
pwsh -File scripts/run_browser_intent_smoke.ps1
```

## Edge 本地插件 MVP

`edge-extension/` 是 Manifest V3 插件：采集导航与可选的本地历史摘要，先脱敏，再只请求 `127.0.0.1` 的预测服务；`local_predictor/server.py` 是无模型调用的确定性 heuristic baseline，用于先验证插件协议和隐私边界。

```powershell
python local_predictor/server.py
# 然后在 Edge 的 edge://extensions 中加载 edge-extension/（开发者模式）
```

插件需要 `history`、`tabs`、`webNavigation`、`storage`、`sidePanel` 权限；关闭 `collectHistory` 后只使用当前 observation。仓库不包含任何真实浏览历史、用户配置、模型缓存或绝对路径。离线插件测试结果见 [`results/browser_intent_extension_smoke.json`](results/browser_intent_extension_smoke.json)。

当前状态：源码、脱敏测试、本地服务健康检查和 HTTP 预测已验证；Edge 真机加载仍需在本机手动打开 `edge://extensions/` 后选择上述目录。自动化浏览器策略不允许代理访问该内部管理页，因此未把未发生的 UI 安装写成已完成。

## 当前结论（仅限 smoke）

- NanoJev + Qwen3-0.6B 在本机原生 Windows CUDA 路径可完成 Decision Head 训练和长轨迹无解码推理。
- v2 250 条数据的审计通过，但当前独立留出样本仍太小，且 smoke 置信度接近 1；不能据此宣称“0.6B 已适合常驻 Browser semantic layer”。
- 同一 v2 上的 `heuristic-v0` 已达到 96% 当前意图准确率，说明模板中的标题/元素语义仍然很强；后续必须加入跨域、模糊标题和人工会话标签，否则这个分数不能代表真实用户预测能力。
- 更严格的 lexical ablation 把 v2 checkpoint 的准确率从 100% 降到 0%，确认当前合成 benchmark 存在 shortcut；诊断细节见 [`results/browser_intent_model_diagnostics.json`](results/browser_intent_model_diagnostics.json)。
- Decider 的协议层测试通过；其 2B 权重本次下载未完成，因此没有伪造 decider 的本机精度。
- 要回答“是否适合常驻”，下一阶段必须扩大人工/半真实标签、按用户会话而非页面随机切分，并完成校准集、长轨迹退化曲线和 Edge 真机安装验证。
