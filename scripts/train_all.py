#!/usr/bin/env python3
"""Unified training orchestrator -- trains all VLA modules sequentially."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

_LOG = logging.getLogger(__name__)

_SCRIPTS = {
    "perception": "scripts/train_perception.py",
    "hoi": "scripts/train_hoi.py",
    "hazard": "scripts/train_hazard_vlm.py",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all VLA modules sequentially.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default="configs/backend_mps.yaml")
    p.add_argument("--modules", nargs="+", default=["perception", "hoi", "hazard"],
                    choices=list(_SCRIPTS.keys()))
    p.add_argument("--output-dir", default="trained_models")
    p.add_argument("--epochs", type=int, default=None, help="Override per-module epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override per-module batch size.")
    return p.parse_args()


def _run_module(
    module: str,
    args: argparse.Namespace,
) -> dict:
    script = _SCRIPTS[module]
    cmd = [
        sys.executable, script,
        "--config", args.config,
        "--backend-config", args.backend_config,
        "--output", str(Path(args.output_dir) / module / "checkpoint.pt"),
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]

    _LOG.info("\n%s", "=" * 60)
    _LOG.info("  Training module: %s", module)
    _LOG.info("  Script: %s", script)
    _LOG.info("%s\n", "=" * 60)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - t0

    return {
        "module": module,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    args = parse_args()
    summary: list[dict] = []

    for module in args.modules:
        if module not in _SCRIPTS:
            _LOG.info("[train_all] Unknown module '%s', skipping.", module)
            continue
        result = _run_module(module, args)
        summary.append(result)
        if result["returncode"] != 0:
            _LOG.warning("[train_all] WARNING: %s exited with code %s", module, result["returncode"])

    report_path = Path(args.output_dir) / "training_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _LOG.info("\n[train_all] Summary written to %s", report_path)

    for r in summary:
        status = "OK" if r["returncode"] == 0 else f"FAILED({r['returncode']})"
        _LOG.info("  %12s  %s  %.1fs", r["module"], status, r["elapsed_seconds"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
