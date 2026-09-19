# Mind2Web 本地推理对比技术报告

**实验日期：** 2026-09-19  
**实验平台：** 原生 Windows + CUDA  
**GPU：** NVIDIA GeForce RTX 4070 Ti SUPER 16 GB  
**对比对象：** NanoJev（Qwen3-0.6B 基座）与公开 Decider-2B 权重  
**实验目的：** 使用公开的 Mind2Web 测试集，比较两个模型在网页候选元素选择任务上的准确率、长跨网站泛化能力和本地推理速度。

## 1. 结论先行

在本次 Mind2Web 候选元素选择实验中：

- Decider-2B 的 Top-1 准确率为 **77.45%**；
- NanoJev 的 Top-1 准确率为 **30.91%**；
- Decider-2B 的 Top-3 准确率为 **95.22%**，NanoJev 为 **64.16%**；
- NanoJev 的 P50 延迟为 **27.0 ms**，Decider-2B 为 **103.4 ms**；
- NanoJev 的吞吐量约为 **36.45 条/秒**，Decider-2B 为 **9.68 条/秒**；
- Decider-2B 使用了更多推理显存，但仍可在本机 16 GB 显卡上运行。

因此，在这个公开网页动作选择任务上：

> **Decider-2B 明显更准确，NanoJev 明显更快。**

这不是完整浏览器自动执行成功率测试，而是“给定任务、最近动作和 6 个候选元素，选择正确目标元素”的离线测试。

## 2. Mind2Web 数据与隐私处理

Mind2Web 是一个真实网站网页代理数据集，官方介绍包含超过 2,000 个任务、137 个网站和 31 个领域。官方测试划分为 Cross-Task、Cross-Website 和 Cross-Domain。官方仓库还特别要求测试文件不要重新分发，也不要进入训练语料。

本实验只下载官方测试压缩包，没有下载训练集，也没有把 Mind2Web 数据加入任何模型训练。官方压缩包 SHA-256 为：

```text
8f5fbe72afab942fe97cdf7fb397e179885d89b5c16862288e9a14bc6d41ca89
```

原始压缩包和解压后的 HTML 只保存在本机 `work/mind2web/raw/`，该目录被 Git 忽略，不会提交或推送。

### 2.1 输入转换

为了匹配 typed decision 模型的能力，原始动作被转换为：

```text
网站和领域
用户任务描述
最近最多 4 个已完成动作
下一步操作类型
6 个候选网页元素
```

候选集按照 Decider 官方 Mind2Web 数据加载方式构建：

- 1 个正样本元素；
- 5 个负样本元素；
- 候选元素只保留 tag 和属性描述；
- 原始 HTML 不送入模型；
- 只保留候选元素文本和结构信息。

## 3. 测试规模

| 官方 split | 任务数 | 动作状态数 | 网站数 | 领域数 |
|---|---:|---:|---:|---:|
| Cross-Task | 252 | 1,974 | 69 | 3 |
| Cross-Website | 177 | 1,314 | 10 | 3 |
| Cross-Domain | 912 | 5,590 | 54 | 2 |
| 合计 | 1,341 | 8,878 | — | — |

动作类型分布：

| 操作 | 状态数 |
|---|---:|
| CLICK | 7,405 |
| TYPE | 1,170 |
| SELECT | 303 |

## 4. 实验方法

两个模型使用完全相同的准备后输入：

- 原生 Windows CUDA；
- BF16；
- 单条状态逐条调用；
- 预热 5 条；
- 模型加载时间不计入延迟；
- 每次推理前后同步 CUDA；
- 输入上限统一为 8,192 tokens；
- 没有网络请求；
- 没有在 Mind2Web 上重新训练 NanoJev 或 Decider-2B。

本次比较不是严格的同训练数据零样本比较：当前 NanoJev 检查点没有用 Mind2Web 训练；Decider-2B 使用公开权重，而其公开代码包含 Mind2Web 训练任务加载器，因此不能排除公开权重曾接触过 Mind2Web 训练 split。无论如何，本地实验没有使用 Mind2Web 测试 split 做训练或微调。

准确率指标：

- **Top-1**：最高概率候选是否为正确元素；
- **Top-3**：正确元素是否在前三个候选中；
- **宏平均步骤准确率**：先按任务计算步骤准确率，再对任务取平均；
- **固定候选序列全步骤准确率**：一个任务的所有已评估动作是否全部选对。这不是实际浏览器执行成功率，只是离线诊断指标。

## 5. 总体结果

| 指标 | NanoJev | Decider-2B |
|---|---:|---:|
| 动作状态数 | 8,878 | 8,878 |
| Top-1 准确率 | 30.91% | **77.45%** |
| Top-3 准确率 | 64.16% | **95.22%** |
| 宏平均步骤准确率 | 30.43% | **77.95%** |
| 固定候选序列全步骤准确率 | 1.72% | **30.95%** |
| 总推理时间 | **243.58 秒** | 917.22 秒 |
| 吞吐量 | **36.45 条/秒** | 9.68 条/秒 |
| 平均延迟 | **27.42 ms** | 103.30 ms |
| P50 延迟 | **27.04 ms** | 103.38 ms |
| P95 延迟 | **35.43 ms** | 109.22 ms |
| 最大单条延迟 | **44.71 ms** | 121.91 ms |
| 推理峰值 allocated | **2,363.66 MB** | 3,676.13 MB |
| 推理峰值 reserved | **2,404.00 MB** | 3,726.00 MB |

Decider-2B 的 Top-1 比 NanoJev 高 **46.54 个百分点**，但 P50 延迟约为 NanoJev 的 **3.8 倍**。NanoJev 的吞吐量约为 Decider-2B 的 **3.8 倍**。

## 6. 不同泛化设置

| Split | NanoJev Top-1 | Decider-2B Top-1 | NanoJev 宏平均 | Decider-2B 宏平均 |
|---|---:|---:|---:|---:|
| Cross-Task | 34.85% | **83.59%** | 33.75% | **82.98%** |
| Cross-Website | 25.27% | **73.21%** | 25.23% | **74.21%** |
| Cross-Domain | 30.84% | **76.28%** | 30.53% | **77.28%** |

两个模型在未见过的网站和领域上都会下降，但 Decider-2B 仍保持较高的候选选择能力。NanoJev 在 Cross-Website 上只有约 25% 的 Top-1，说明当前 NanoJev 检查点从合成浏览意图迁移到真实网页动作选择时，泛化能力有限。

## 7. 不同操作类型

| 操作 | NanoJev Top-1 | Decider-2B Top-1 |
|---|---:|---:|
| CLICK | 29.98% | **73.38%** |
| TYPE | 35.38% | **98.89%** |
| SELECT | 36.30% | **94.06%** |

Decider-2B 在 TYPE 和 SELECT 上尤其稳定。需要注意，当前上下文显式包含下一步操作类型，因此这不是纯粹的“只看页面观察预测动作”实验；它更准确地表示“已知下一步操作类型时选择目标元素”。

## 8. 对两个模型的解释

### NanoJev

NanoJev 的优势是：

- 单条延迟低；
- 吞吐量高；
- 显存占用较低；
- 更适合浏览器插件中的高频候选预筛选。

主要问题是：

- 当前检查点不是用 Mind2Web 训练的；
- 训练目标是合成浏览器意图类别，而不是网页元素选择；
- 在真实网页候选元素上 Top-1 只有 30.91%；
- Cross-Website 泛化明显下降。

### Decider-2B

Decider-2B 的优势是：

- 更适合 typed Choice 形式的候选选择；
- 在三个 Mind2Web split 上都明显高于 NanoJev；
- Top-3 达到 95.22%，适合后续由程序进行二次验证或重排；
- 在未见网站和未见领域上仍保持较好表现。

主要代价是：

- 单条延迟约 103 ms；
- 吞吐量约为 NanoJev 的四分之一；
- 显存和模型权重更大；
- 仍然没有证明可以直接完成真实浏览器任务。

## 9. 与当前 Edge 实验的关系

Mind2Web 和本机 Edge 数据不能直接合并成一个准确率：

- Edge 数据测试的是个人浏览轨迹和下一页类别预测；
- Mind2Web 测试的是给定任务后选择网页动作目标；
- Edge 数据没有可靠的用户真实意图标签；
- Mind2Web 有明确的动作目标，但不等于用户启动浏览器时的潜在意图。

两者结合后的工程判断是：

```text
Edge 历史先验
    ↓
预测可能的任务/页面类别
    ↓
Decider-2B 选择候选网页元素或下一步动作
    ↓
浏览器代码验证并决定是否执行
```

当前更合理的架构是让 NanoJev 做低延迟预筛选，让 Decider-2B 处理需要更高可靠性的候选选择；两者都不应在没有代码验证和升级路径的情况下直接执行不可逆操作。

## 10. 限制与下一步

本实验不是完整 WebArena 式在线任务执行，原因是：

1. 没有启动 Mind2Web 在线网站环境；
2. 没有执行真实点击、输入和页面跳转；
3. 候选元素使用 1 正样本 + 5 负样本，而不是页面上的全部 DOM 元素；
4. 输入去除了原始 HTML，只保留了候选元素的结构属性；
5. Decider-2B 使用公开权重，没有在本机重新训练；
6. NanoJev 也没有在 Mind2Web 上微调；
7. 两个模型的公开训练历史并不完全相同，因此准确率差异不能全部归因于参数量。

因此，本报告可以证明：

> 在相同的 Mind2Web 候选集合上，Decider-2B 的离线目标元素选择能力明显强于当前 NanoJev；NanoJev 的本地响应速度和吞吐量明显更好。

本报告不能证明：

> 任一模型已经可以独立完成真实网站上的完整任务。

下一步最有价值的实验是只使用 Mind2Web 训练 split 对 Decider-2B 做 LoRA/QLoRA 适配，再使用这三个官方测试 split 对比微调前后准确率、长序列表现、ECE 和延迟。

## 11. 复现实验入口

原始数据和准备后的测试输入都在 `work/` 下，不应提交到公开仓库。代码入口为：

```powershell
$py = ".\\.venv-win-ar\\Scripts\\python.exe"
$deciderModel = (Get-ChildItem .hf-cache-win/hub/models--Mapika--decider-2b/snapshots -Directory | Select-Object -First 1).FullName

& $py scripts/prepare_mind2web_benchmark.py `
  --test-root work/mind2web/raw/test `
  --output-dir work/mind2web/prepared-full `
  --limit-per-split 0

& $py scripts/benchmark_mind2web_inference.py `
  --model nanojev `
  --input-dir work/mind2web/prepared-full `
  --output work/mind2web/results/nanojev-full.json `
  --predictions work/mind2web/results/nanojev-full.jsonl `
  --nanojev-root .tools/NanoJev `
  --nanojev-checkpoint work/browser-intent-large/nanojev `
  --decider-root .tools/decider `
  --decider-model $deciderModel `
  --warmup 5 `
  --max-length 8192
```

将 `--model nanojev` 改为 `--model decider`，并将 `--max-length` 改为 `--max-state-tokens 8192`，可运行 Decider-2B。

官方资料：

- https://github.com/OSU-NLP-Group/Mind2Web
- https://arxiv.org/abs/2306.06070
- https://huggingface.co/datasets/osunlp/Mind2Web
