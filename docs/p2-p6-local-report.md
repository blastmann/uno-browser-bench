# P2–P6 本地验证报告入口

本文件说明本地验证入口。包含 Edge 个人历史的精确统计、模型预测和校准结果的文件均放在 `work/`，并由 `.gitignore` 忽略；不要把这些文件推送到远程仓库。

## 可重复命令

```powershell
python scripts/build_decision_dataset.py `
  --input data/browser_intent/observation_v2.jsonl `
  --output-dir work/browser-intent-v3/decisions `
  --prefixes 5,10,20,50,100,200

python scripts/audit_decision_splits.py `
  --input-dir work/browser-intent-v3/decisions `
  --output work/browser-intent-v3/split_audit.json

python scripts/calibrate_decider_and_curve.py `
  --predictions work/browser-intent-v3/decider-v3-heldout.json `
  --gold-dir work/browser-intent-v3/decisions `
  --output work/browser-intent-v3/calibration_curve.json

python scripts/grade_browser_intent.py `
  --split-audit work/browser-intent-v3/split_audit.json `
  --edge-audit work/edge_trace_v3_local_audit.json `
  --calibration-report work/browser-intent-v3/calibration_curve.json `
  --model-diagnostics results/browser_intent_model_diagnostics.json `
  --output work/browser-intent-v3/p2-p6_grade.json
```

## 判定原则

- P2 同时检查 JSON/合同、state id、重复输入、冲突标签、family 跨 split 泄漏，以及 Edge trace 输出的隐私边界。
- P3 只在独立 calibration split 上拟合一个温度，再在 test/OOD 上报告 ECE、NLL、Brier 和 risk-coverage。
- P4 固定报告 5/10/20/50/100/200 events，不能用总体平均替代曲线。
- P6 只有在真实/半真实人工标签、校准和长轨迹结果都达到门槛后，才能宣布适合常驻；否则最多做候选排序和升级路由。

注意：合成数据的 completed-task label 与 Edge 历史的 observed-next-category 都不是用户潜在意图的直接真值。
