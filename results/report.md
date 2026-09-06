# Uno Browser Bench 实测报告

生成时间：2026-09-06 23:04:34

## 结论摘要

- 在同一份实际输入（552/1432/4941/9627 tokens，目标为 256/1K/4K/8K）和 128-token 输出下，原生 Windows Uno Linear 相对 AR 的 TPS speedup 为 **4.56×**（3.77–4.92×）；WSL2 为 **4.69×**（3.87–5.07×）。
- 原生 Windows / WSL2 的同路径 TPS 比例：AR 为 **1.03–1.06**，Uno 为 **1.01–1.02**；本次实测没有看到 WSL2 的明显性能损耗，差异在约 1–6% 范围内。
- 原生 Windows Uno Linear 已真实完成 4/4 speed 档位和 100 项语义质量任务；通过 Windows-only Gloo process group 兼容补丁运行，Linux/WSL 默认仍使用 NCCL。
- 原生 Uno 语义质量：JSON valid **73.0%**，schema valid **39.0%**，实体 F1 **7.7%**，intent accuracy **0.0%**，step F1 **0.0%**。

## 环境与安装

- Windows 主机：本地 Windows 测试主机；GPU：NVIDIA GeForce RTX 4070 Ti SUPER；显存：16,376 MiB。
- Windows：Microsoft Windows 11 专业版 build 26200；驱动：32.0.16.1088。
- 原生 Python：独立 `.venv-win-ar`，CPython 3.10.16；PyTorch 2.11.0+cu128；Transformers 5.16.1；einops 0.8.2。
- 原生 CUDA：Toolkit 12.8.1，nvcc 12.8.93；Visual Studio/MSVC 2022 14.44.35207。
- 原生 FlashAttention：2.8.3 从源码针对 torch 2.11.0+cu128 构建，并通过 BF16 CUDA kernel smoke；`triton-windows==3.8.0.post28`。
- WSL2 主路径：Uno 官方 Linux 路径，Python 3.10、PyTorch cu128、FA2、NCCL；Uno Linear 未引入 FA3/Tree sampler。
- 官方兼容性说明：Uno/FlashAttention 的官方主路径仍偏向 Linux；Windows 结果是本机已验证的兼容构建，不等于上游对 Windows 的正式支持。

## WSL2：官方路径 speed A/B

| Target tokens | Actual tokens | AR TPS | Uno TPS | Speedup | AR P50/P95 s | Uno P50/P95 s | AR TPF | Uno TPF | Uno peak VRAM MB | AR/Uno GPU util % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 552 | 46.4 | 222.2 | 4.79x | 2.709/3.110 | 0.576/0.597 | 1.00 | 1.52 | 13237 | 39/9 |
| 1024 | 1432 | 45.4 | 228.8 | 5.04x | 2.807/3.171 | 0.561/0.583 | 1.00 | 1.60 | 13241 | 44/41 |
| 4096 | 4941 | 43.0 | 166.6 | 3.87x | 2.925/3.356 | 0.765/0.797 | 1.00 | 1.34 | 13126 | 42/14 |
| 8192 | 9627 | 38.5 | 195.4 | 5.07x | 3.276/3.749 | 0.653/0.670 | 1.00 | 1.81 | 12947 | 58/50 |

## 原生 Windows：FA2 + Uno Linear 对齐 speed A/B

| Target tokens | Actual tokens | AR TPS | Uno TPS | Speedup | AR P50/P95 s | Uno P50/P95 s | AR TPF | Uno TPF | Uno peak VRAM MB | AR/Uno GPU util % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 552 | 47.8 | 225.6 | 4.71x | 2.674/2.690 | 0.565/0.586 | 1.00 | 1.52 | 13272 | 46/96 |
| 1024 | 1432 | 47.3 | 232.8 | 4.92x | 2.703/2.735 | 0.553/0.568 | 1.00 | 1.60 | 13324 | 47/96 |
| 4096 | 4941 | 44.8 | 169.1 | 3.77x | 2.846/2.903 | 0.752/0.781 | 1.00 | 1.34 | 13324 | 50/96 |
| 8192 | 9627 | 40.9 | 198.0 | 4.83x | 3.125/3.144 | 0.644/0.661 | 1.00 | 1.81 | 13333 | 55/97 |

## 原生 Windows / WSL2 同路径比较

| Target tokens | Actual tokens | Native AR TPS | WSL AR TPS | Native/WSL AR | Native Uno TPS | WSL Uno TPS | Native/WSL Uno |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 552 | 47.8 | 46.4 | 1.032 | 225.6 | 222.2 | 1.015 |
| 1024 | 1432 | 47.3 | 45.4 | 1.042 | 232.8 | 228.8 | 1.018 |
| 4096 | 4941 | 44.8 | 43.0 | 1.042 | 169.1 | 166.6 | 1.015 |
| 8192 | 9627 | 40.9 | 38.5 | 1.063 | 198.0 | 195.4 | 1.013 |

注：这是同一台机器、同一模型/输入生成方式、同一 FA2 backend 的观测对照，不是跨机器或严格 OS 微基准；1–6% 差异可能包含运行时、调度和测量噪声。

## 语义质量（50 条数据，实体+行为共 100 项）

| runner | JSON valid | schema valid | entity F1 | intent acc | step F1 |
|---|---:|---:|---:|---:|---:|
| WSL AR | 44.0% | 17.0% | 4.3% | 0.0% | 0.0% |
| WSL Uno | 73.0% | 40.0% | 8.7% | 0.0% | 0.0% |
| Native AR FA2 | 72.0% | 46.0% | 12.0% | 0.0% | 0.0% |
| Native Uno Linear | 73.0% | 39.0% | 7.7% | 0.0% | 0.0% |

### Uno 长轨迹 JSON valid（原生 Windows）

| events | valid |
|---:|---:|
| 5 | 100.0% |
| 10 | 100.0% |
| 20 | 62.5% |
| 50 | 50.0% |
| 100 | 62.5% |
| 200 | 56.2% |

## 解释与限制

- 输入长度表同时记录目标档位和 tokenizer 实际长度；当前合成 filler 实际为 552/1432/4941/9627 tokens，因此满足至少 256/1K/4K/8K 的要求。
- Uno TPF 是官方 decoder 的 `accepts / forwards`；AR 固定为 1.0。P50/P95 是每档 30 次正式 run 的总生成 latency；Uno 通过 completion callback 的请求间隔近似得到。
- TTFT：语义质量 runner 对 AR 记录了首次 logits processor 时间；Uno 官方公开 generate API 只暴露完成回调，没有逐请求首 token回调，因此 speed JSON 不伪造 Uno TTFT，原始质量 JSON 保留 `ttft_s: null` 及该限制。
- 质量数据是可重复的合成/半真实 Browser Observation，不含原始 HTML，字段覆盖 url/title/elements/events，域包括电商、搜索、新闻、GitHub、邮件、社交、后台和噪声页；50 条 smoke 集和扩展脚本均保留。
- 原生 Windows Uno 兼容证据：Native Uno Linear is validated through a documented Windows-only Gloo compatibility path. The patched process-group backend is required because native Windows PyTorch has no NCCL; the Uno Linear engine, source-built FlashAttention, speed benchmark, and semantic smoke all completed successfully.

## 三个问题的回答

1. **Uno 是否明显更快？** 是。对齐后的真实 token 测量中，原生 Windows speedup 平均 4.56×，WSL2 平均 4.69×。
2. **0.9B 对实体/行为语义是否够用？** 作为粗粒度 JSON parser 有一定能力，但实体 F1 只有约 8–12%，intent/step 严格准确率为 0%；不适合未经约束解码、后处理或蒸馏就直接作为生产常驻 semantic layer。
3. **长轨迹是否快速退化？** 是。原生 Uno 的 trajectory JSON valid 在 5/10 events 为 100%，20 events 约 62.5%，50 events 50%，100 events 62.5%，200 events 56.3%；长轨迹应先做事件压缩/分段摘要。

## 来源

- Uno 官方仓库：https://github.com/ifm-ai/uno
- K2-Horizon-0.9B-Uno：https://huggingface.co/IFM/K2-Horizon-0.9B-Uno
- K2-Horizon-0.9B：https://huggingface.co/IFM/K2-Horizon-0.9B
- FlashAttention：https://github.com/Dao-AILab/flash-attention
- CUDA 12.8 Windows 安装文档：https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-microsoft-windows/
- WSL GPU compute：https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
