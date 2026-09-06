param(
    [int]$Warmup = 5,
    [int]$Runs = 30,
    [string]$InputLengths = "256,1024,4096,8192",
    [double]$GpuMemoryUtilization = 0.90
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv-win-ar\Scripts\python.exe"
$Cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_HOME = $Cuda
$env:CUDA_PATH = $Cuda
$env:Path = "$Cuda\bin;$env:Path"
$env:HF_HOME = Join-Path $Root ".hf-cache-win"
$env:HF_HUB_DISABLE_XET = "1"
$env:TRITON_CACHE_DIR = Join-Path $Root ".triton-cache-win"
$env:NANO_VLLM_DIST_BACKEND = "gloo"

if (-not (Test-Path $Python)) {
    throw "Native Windows environment not found: $Python"
}

& $Python (Join-Path $Root "scripts\run_speed_bench.py") `
    --mode uno `
    --output (Join-Path $Root "results\speed_uno_native_fixed_tokens.jsonl") `
    --input-lengths $InputLengths `
    --output-tokens 128 `
    --warmup $Warmup `
    --runs $Runs `
    --gpu-memory-utilization $GpuMemoryUtilization 2>&1 | Tee-Object (Join-Path $Root "logs\speed_uno_native_fixed_tokens.log")
