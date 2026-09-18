param(
  [string]$Model = "Qwen/Qwen3-0.6B",
  [int]$Steps = 30,
  [int]$HeadSteps = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv-win-ar\Scripts\python.exe"
$NanoJev = Join-Path $Root ".tools\NanoJev\scripts"
$Input = Join-Path $Root "data\browser_intent\observation_v2.jsonl"
$DecisionDir = Join-Path $Root "work\browser-intent-smoke\decisions"
$OutputDir = Join-Path $Root "work\browser-intent-smoke\nanojev"

if (!(Test-Path $Python)) { throw "Missing local Python environment: .venv-win-ar" }
if (!(Test-Path (Join-Path $NanoJev "train_pipeline_decisions.py"))) { throw "Missing local NanoJev checkout under .tools/NanoJev" }

& $Python (Join-Path $Root "scripts\build_decision_dataset.py") --input $Input --output-dir $DecisionDir --prefixes 5
& $Python (Join-Path $Root "scripts\validate_decision_contract.py") --input-dir $DecisionDir --output (Join-Path $DecisionDir "contract.json")

$env:PYTHONPATH = $NanoJev
& $Python (Join-Path $NanoJev "train_pipeline_decisions.py") `
  --input $DecisionDir --output-dir $OutputDir --model $Model `
  --objective observed_outcome --loss ce --steps $Steps --head-steps $HeadSteps `
  --batch-questions 1 --microbatch-questions 1 --max-length 512 --precision bf16 --eval-every 10

& $Python (Join-Path $Root "scripts\evaluate_decision_predictions.py") `
  --input (Join-Path $OutputDir "predictions_test.jsonl") `
  --output (Join-Path $OutputDir "typed_metrics_test.json")
