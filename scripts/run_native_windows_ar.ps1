param(
    [int]$Warmup = 5,
    [int]$Runs = 30,
    [string]$InputLengths = "256,1024,4096,8192"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv-win-ar\Scripts\python.exe"
$Cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_HOME = $Cuda
$env:CUDA_PATH = $Cuda
$env:Path = "$Cuda\bin;$env:Path"
$Cache = Join-Path $Root ".hf-cache-win"
$env:HF_HOME = $Cache
$env:HF_HUB_DISABLE_XET = "1"

if (-not (Test-Path $Python)) {
    throw "Native Windows AR environment not found: $Python"
}

& $Python (Join-Path $Root "scripts\run_speed_bench.py") `
    --mode ar `
    --output (Join-Path $Root "results\speed_ar_native_fa2_aligned_tokens.jsonl") `
    --input-lengths $InputLengths `
    --output-tokens 128 `
    --warmup $Warmup `
    --runs $Runs `
    --attn-implementation flash_attention_2 2>&1 | Tee-Object (Join-Path $Root "logs\speed_ar_native_fa2_aligned_tokens.log")
