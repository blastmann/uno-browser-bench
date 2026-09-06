#!/usr/bin/env python3
"""Collect a JSON environment record from inside WSL."""
from __future__ import annotations
import argparse, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

def run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as e:
        return f"ERROR: {e}"

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(), "kernel": platform.release(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": run("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,pstate,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"),
        "wsl_gpu": run("nvidia-smi -L"), "nvcc": run("nvcc --version"),
        "free_memory": run("free -h"), "disk": run("df -h / /mnt/c"),
        "torch": None, "packages": {},
    }
    try:
        import torch
        record["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
        if torch.cuda.is_available(): record["torch"]["device"] = torch.cuda.get_device_name(0)
    except Exception as e: record["torch"] = {"error": repr(e)}
    for pkg in ("transformers", "peft", "flash_attn", "safetensors"):
        record["packages"][pkg] = run(f"{shutil.which('python') or sys.executable} -c \"import {pkg}; print(getattr({pkg}, '__version__', 'present'))\"")
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(record, ensure_ascii=False))
if __name__ == "__main__": main()
