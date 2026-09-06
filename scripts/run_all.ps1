param(
  [int]$QualityLimit = 50,
  [int]$Warmup = 5,
  [int]$Runs = 30,
  [switch]$SkipSpeed
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$WslRoot = (wsl wslpath -a ($Root -replace '\\','/')).Trim()
$WslHome = (wsl -d Ubuntu -- bash -lc 'printf %s "$HOME"').Trim()
$EnvName = 'uno-browser-bench'
$Conda = "$HOME/miniconda3/bin/conda"
wsl -d Ubuntu -- bash -lc "mkdir -p '$WslRoot/logs' '$WslRoot/results' '$WslRoot/data'"
wsl -d Ubuntu -- bash -lc "source '$HOME/miniconda3/etc/profile.d/conda.sh'; conda activate $EnvName; python '$WslRoot/scripts/generate_dataset.py' --output '$WslRoot/data/browser_semantics.jsonl' --count 200 --seed 20260906; python '$WslRoot/scripts/measure_environment.py' --output '$WslRoot/results/environment.json'" *>&1 | Tee-Object "$Root\logs\dataset_environment.log"
if (-not $SkipSpeed) {
  wsl -d Ubuntu -- bash -lc "source '$WslHome/miniconda3/etc/profile.d/conda.sh'; conda activate uno-ar; python '$WslRoot/scripts/run_speed_bench.py' --mode ar --output '$WslRoot/results/speed_ar_fixed.jsonl' --warmup $Warmup --runs $Runs --input-lengths 256,1024,4096,8192" *>&1 | Tee-Object "$Root\logs\speed_ar_fixed.log"
  foreach ($length in @(256,1024,4096,8192)) {
    wsl -d Ubuntu -- bash -lc "source '$WslHome/miniconda3/etc/profile.d/conda.sh'; conda activate $EnvName; python '$WslRoot/scripts/run_speed_bench.py' --mode uno --output '$WslRoot/results/speed_uno_fixed_${length}.jsonl' --warmup $Warmup --runs $Runs --input-lengths $length" *>&1 | Tee-Object "$Root\logs\speed_uno_fixed_${length}.log"
  }
}
wsl -d Ubuntu -- bash -lc "source '$WslHome/miniconda3/etc/profile.d/conda.sh'; conda activate uno-ar; python '$WslRoot/scripts/run_model_benchmark.py' --mode ar --dataset '$WslRoot/data/browser_semantics.jsonl' --output '$WslRoot/results/ar_semantics_fixed_50.jsonl' --task all --limit $QualityLimit --batch-size 2 --max-new-tokens 256" *>&1 | Tee-Object "$Root\logs\quality_ar_fixed.log"
wsl -d Ubuntu -- bash -lc "python3 '$WslRoot/scripts/evaluate_outputs.py' --input '$WslRoot/results/ar_semantics_fixed_50.jsonl' --output '$WslRoot/results/ar_quality_fixed_50.json'" *>&1 | Tee-Object "$Root\logs\quality_ar_fixed_eval.log"
wsl -d Ubuntu -- bash -lc "source '$WslHome/miniconda3/etc/profile.d/conda.sh'; conda activate $EnvName; python '$WslRoot/scripts/run_model_benchmark.py' --mode uno --dataset '$WslRoot/data/browser_semantics.jsonl' --output '$WslRoot/results/uno_semantics_fixed_50.jsonl' --task all --limit $QualityLimit --batch-size 4 --max-new-tokens 256" *>&1 | Tee-Object "$Root\logs\quality_uno_fixed.log"
wsl -d Ubuntu -- bash -lc "python3 '$WslRoot/scripts/evaluate_outputs.py' --input '$WslRoot/results/uno_semantics_fixed_50.jsonl' --output '$WslRoot/results/uno_quality_fixed_50.json'" *>&1 | Tee-Object "$Root\logs\quality_uno_fixed_eval.log"
wsl -d Ubuntu -- bash -lc "python3 '$WslRoot/scripts/summarize_results.py' --root '$WslRoot'" *>&1 | Tee-Object "$Root\logs\summarize.log"
Write-Host "Completed. Results under $Root\results"
