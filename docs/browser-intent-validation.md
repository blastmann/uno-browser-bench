# Browser Intent 验证路线

## 目标

验证一个小模型是否能在本机 Edge 的 Browser Observation trace 上完成：

1. 当前轨迹意图分类：`intent`。
2. 页面实体和属性抽取：受约束 typed decision 或 JSON。
3. 行为序列摘要：`steps` / `sequence_summary`。
4. 浏览器启动时的下一意图预测：`next_intent`。
5. 置信度校准：模型的概率能否支持“执行 / 观察 / 转交大模型”的门控。

## 固定实验协议

- Observation 只包含 `url/title/elements/events`，不包含原始 HTML。
- 数据切分按 `template_id` 和用户会话切分，不按随机页面行切分。
- 长度档位固定为 5、10、20、50、100、200 events。
- 分类报告至少包括 accuracy、macro-F1、NLL、Brier、ECE、risk-coverage。
- 实体报告包括 JSON valid、schema valid、entity P/R/F1、核心字段 exact match。
- 任何训练失败、下载失败、显存不足或依赖不兼容都保留原始原因，不用估算值替代。

## 模型阶梯

1. `heuristic-v0`：本地协议和隐私边界 baseline，不作为模型结论。
2. NanoJev / Qwen3-0.6B：首选 `<1B` typed-decision 实验，先做 Linear/attention-only 路径。
3. 普通 Qwen3-0.6B：同数据、同 prompt 的生成/分类 baseline。
4. decider / Qwen3.5-2B：质量控制上限，不是常驻目标；比较 typed probabilities、NLL/ECE 和长序列性能。
5. 只有在小模型达标且校准稳定后，才考虑量化和 Edge 常驻部署。

## 通过门槛

- P0：数据审计无解析错误、重复输入和 label leakage；合同校验通过。
- P1：至少一个模型在原生 Windows CUDA 完成训练、保存 checkpoint、加载推理并产出逐样本结果。
- P2：test 与 OOD 均需报告，且 OOD 不能只来自相同页面模板的重复样本。
- P3：ECE 与 selective accuracy 需要在独立 calibration split 上拟合温度后再评估。
- P4：长轨迹退化必须按长度绘制，不能只报告总体平均。
- P5：Edge 插件只向 localhost 发送脱敏 observation；源码测试、health check、HTTP prediction 和真机安装分别验收。
- P6：最终结论分为“适合常驻 / 适合候选排序但需升级 / 不适合”，不得由 smoke 结果直接外推。

## 目前完成度

- P0：已完成，见 `data/browser_intent/audit_v2.json` 和 `data/browser_intent/contract_v2.json`。
- P1：NanoJev smoke 已完成；原生 Windows CUDA 可执行，见 `results/browser_intent_smoke_summary.json`。
- P1：decider 协议单元测试和公开 2B 权重本机 CUDA 推理已完成；公开 v2 test+OOD 为 68 条、整体意图准确率 92.65%，lexical ablation 为 0%，结果见 `results/browser_intent_model_diagnostics.json`。
- P5：插件离线测试、localhost heuristic 服务、Edge 扩展加载和导航触发已完成；localhost 收到的运行证据不含原始 URL、标题或历史内容。浏览器启动事件仍需单独重启浏览器验证，不能由导航事件替代。
- P2/P3/P4/P6：仍需更大的独立标签集、会话级真实/半真实 trace、温度校准和真机安装后持续运行。
