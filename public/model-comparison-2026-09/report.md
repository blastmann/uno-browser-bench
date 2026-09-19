# Decider-2B、Jev、NanoJev：公开脱敏评测摘要

## 结论

在 Mind2Web 固定候选协议上，Jev 的动作级准确率为 52.02%，Decider-2B 为 38.09%，NanoJev zero-shot 为 7.55%。在 Uno trace-v3 synthetic test 上，Jev 和经过短 prefix 适配训练的 NanoJev 为 100%，Decider-2B 为 89.13%。

NanoJev 的 trace-v3 结果来自本地 synthetic 数据适配训练，不是 zero-shot 结果；因此不能与 Jev/Decider-2B 做严格同条件排名。

## Mind2Web `test_domain`

### 协议

- 1,000 个固定协议样本，其中 940 个有可用正例金标准。
- 候选上限 50；没有候选生成器 rank 文件。
- 统计对象是固定候选集下的动作目标选择，不宣称复现官方 Candidate Recall@50。
- 原始网页状态、HTML、URL、候选文本和 DOM 标识没有包含在本公开导出中。

### 结果

| 指标 | Decider-2B | Jev | NanoJev |
|---|---:|---:|---:|
| 动作级 Accuracy | 38.09% | **52.02%** | 7.55% |
| 任务内动作准确率 macro | 40.28% | **54.36%** | 8.82% |
| 整条任务全部动作正确 | 3.25% | **11.04%** | 0.00% |
| Mean confidence | 0.4066 | **0.5550** | 0.0475 |
| ECE-10 | 0.0376 | 0.0415 | 0.0280* |
| Brier，越低越好 | 0.7654 | **0.6004** | 0.9664 |
| 中位延迟 | 451 ms | 709 ms | 485 ms |
| P95 延迟 | 528 ms | 812 ms | 3,699 ms |

Jev 与 Decider-2B 的配对结果：两者都正确 297、仅 Decider-2B 正确 61、仅 Jev 正确 192、两者都错 390；McNemar 双侧精确检验 `p = 5.78e-17`。

`*` NanoJev 的低 ECE 与其极低平均 confidence 同时出现，不能解释为高质量校准。

## Uno trace-v3

### 协议

- test：138 个 prefix 状态、40 条 source trajectories。
- OOD：138 个 prefix 状态、40 条未见 family trajectories。
- 每个问题 10 个意图候选，prefix 长度为 5、10、20、50、100、200 事件。
- 所有模型使用相同的 canonical JSON 状态和候选集合。
- NanoJev checkpoint 使用 v3 train 中 prefix≤50 的 267 条记录进行适配训练。

### 总体结果

| Split | Decider-2B | Jev | NanoJev |
|---|---:|---:|---:|
| test Accuracy | 89.13% | **100.00%** | **100.00%** |
| test strict trajectory | 85.00% | **100.00%** | **100.00%** |
| OOD Accuracy | 100.00% | 100.00% | 100.00% |
| test Mean confidence | 0.679 | 0.954 | 0.999 |
| test ECE-10 | 0.212 | 0.046 | 0.0005 |
| test 中位延迟 | 46 ms | 538 ms | 83 ms |

### test 按 prefix

| Prefix | 样本数 | Decider-2B | Jev | NanoJev |
|---:|---:|---:|---:|---:|
| 5 | 40 | 100.0% | 100.0% | 100.0% |
| 10 | 32 | 100.0% | 100.0% | 100.0% |
| 20 | 25 | 88.0% | 100.0% | 100.0% |
| 50 | 20 | 70.0% | 100.0% | 100.0% |
| 100 | 14 | 71.4% | 100.0% | 100.0% |
| 200 | 7 | 71.4% | 100.0% | 100.0% |

Decider-2B 的错误集中在一个 repository family；OOD 数据上三个模型都是 100%，因此 OOD 的主要区分信号是置信度，而不是准确率。

## 解释边界

- Mind2Web 的结果是固定候选研究协议，不是官方完整候选排序复现。
- Uno trace-v3 是合成数据，当前难度不足以区分所有模型。
- NanoJev 的 trace 结果是适配训练后结果；NanoJev 的 Mind2Web 结果是公开 checkpoint zero-shot，二者不能混为同一条件。
- 延迟来自不同执行形态：本地 GPU 与 HTTP API，不应直接视为严格成本/吞吐排名。

## 建议

工程上可让本地模型承担低延迟路由，并在低置信度、陌生 family 或 calibration 未覆盖的情况下回退到 Jev/强模型。下一轮应加入意图不足、共享动作、错误动作、噪声和视觉文本冲突等 hard cases。
