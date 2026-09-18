#!/usr/bin/env python3
"""Verify the public, sanitized project artifacts without reading private telemetry.

This is a repository-level audit, not a replacement for GPU or Edge runtime
tests. It makes the remaining external verification boundary explicit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> tuple[int, int]:
    rows = errors = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            errors += 1
    return rows, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/project_verification.json"))
    parser.add_argument("--require-service", action="store_true", help="fail if localhost:8765 is not healthy")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "docs/browser-intent-validation.md",
        "data/browser_intent/observation_v2.jsonl",
        "data/browser_intent/audit_v2.json",
        "data/browser_intent/contract_v2.json",
        "results/browser_intent_smoke_summary.json",
        "results/browser_intent_model_diagnostics.json",
        "results/browser_intent_extension_smoke.json",
        "results/browser_intent_edge_runtime_smoke.json",
        "edge-extension/manifest.json",
        "local_predictor/server.py",
    ]
    missing = [item for item in required if not (root / item).exists()]
    observation_rows, observation_parse_errors = count_jsonl(root / "data/browser_intent/observation_v2.jsonl")
    contract = read_json(root / "data/browser_intent/contract_v2.json")
    audit = read_json(root / "data/browser_intent/audit_v2.json")
    diagnostic = read_json(root / "results/browser_intent_model_diagnostics.json")
    edge_runtime = read_json(root / "results/browser_intent_edge_runtime_smoke.json")
    manifest = read_json(root / "edge-extension/manifest.json")
    personal_tracked = subprocess.run(
        ["git", "ls-files", "results/personal"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    health = {"checked": False, "ok": False, "model": None, "bind": None}
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/healthz", timeout=2) as response:
            body = json.loads(response.read().decode("utf-8"))
            health = {"checked": True, "ok": body.get("ok") is True, "model": body.get("model"), "bind": body.get("bind")}
    except Exception as error:  # service is optional for repository-only verification
        health["error_class"] = type(error).__name__
    checks = {
        "required_files": not missing,
        "observation_jsonl_parse": observation_parse_errors == 0,
        "dataset_audit": not audit.get("parse_errors") and not audit.get("missing_required_fields"),
        "decision_contract": not contract.get("errors"),
        "diagnostics_include_decider": "decider_public_v2" in diagnostic,
        "no_personal_files_tracked": not personal_tracked,
        "extension_host_permission_local_only": manifest.get("host_permissions") == ["http://127.0.0.1:8765/*"],
        "predictor_health_if_running": health["ok"] and health["bind"] == "127.0.0.1",
        "edge_runtime_navigation_verified": edge_runtime.get("extension_runtime", {}).get("loaded") is True
        and edge_runtime.get("extension_runtime", {}).get("requests_observed", 0) >= 1
        and edge_runtime.get("extension_runtime", {}).get("last_request", {}).get("source") == "edge-extension",
        "edge_ui_install_verified": edge_runtime.get("extension_runtime", {}).get("loaded") is True
        and edge_runtime.get("extension_runtime", {}).get("requests_observed", 0) >= 1,
        "edge_startup_runtime_verified": edge_runtime.get("extension_runtime", {}).get("startup_runtime_event_observed") is True,
    }
    repository_checks = {key: value for key, value in checks.items() if key not in {"predictor_health_if_running", "edge_runtime_navigation_verified", "edge_ui_install_verified", "edge_startup_runtime_verified"}}
    service_requirement_pass = (not args.require_service) or checks["predictor_health_if_running"]
    report = {
        "schema_version": "uno-browser-bench-public-verification-v1",
        "checks": checks,
        "summary": {
            "all_repository_checks_pass": all(repository_checks.values()),
            "service_required": args.require_service,
            "service_requirement_pass": service_requirement_pass,
            "edge_runtime_navigation_verified": checks["edge_runtime_navigation_verified"],
            "edge_ui_install_verified": checks["edge_ui_install_verified"],
            "edge_startup_runtime_verified": checks["edge_startup_runtime_verified"],
            "note": "Navigation runtime is verified from a content-free localhost audit. Startup runtime still requires a separate browser restart and is never inferred from navigation.",
        },
        "counts": {
            "observation_rows": observation_rows,
            "observation_parse_errors": observation_parse_errors,
            "contract_rows": contract.get("rows"),
            "contract_errors": contract.get("errors"),
        },
        "predictor_health": health,
        "missing_required_files": missing,
        "tracked_personal_files": personal_tracked,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "checks": checks, "all_repository_checks_pass": report["summary"]["all_repository_checks_pass"], "service_requirement_pass": service_requirement_pass}, ensure_ascii=False))
    if not report["summary"]["all_repository_checks_pass"] or not service_requirement_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
