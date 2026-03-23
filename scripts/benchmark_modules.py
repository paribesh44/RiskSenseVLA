#!/usr/bin/env python3
"""Per-module inference benchmark: FPS, latency, memory for each VLA module."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

_LOG = logging.getLogger(__name__)

from risksense_vla.config import load_config
from risksense_vla.runtime import pick_backend
from risksense_vla.train import benchmark_module

from train_perception import TinyPerceptionModel
from train_hazard_vlm import TinyHazardNet
from risksense_vla.hoi import PredictiveHOINet


_MODULES = {
    "perception": {
        "model_cls": TinyPerceptionModel,
        "dummy_fn": lambda: torch.randn(1, 3, 128, 128),
    },
    "hoi": {
        "model_cls": PredictiveHOINet,
        "dummy_fn": lambda: (torch.randn(1, 256), torch.randn(1, 256)),
    },
    "hazard": {
        "model_cls": TinyHazardNet,
        "dummy_fn": lambda: torch.randn(1, 256),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark VLA module inference.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default=None)
    p.add_argument("--trained-dir", default="trained_models")
    p.add_argument("--modules", nargs="+", default=["perception", "hoi", "hazard"])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--out-dir", default="benchmark_logs")
    return p.parse_args()


def _load_model(name: str, trained_dir: str) -> torch.nn.Module:
    spec = _MODULES[name]
    ckpt = Path(trained_dir) / name / "checkpoint.pt"
    model = spec["model_cls"]()
    if ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        model.load_state_dict(state, strict=False)
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config) if args.backend_config else load_config(args.config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "cpu"))
    device = backend.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    for name in args.modules:
        if name not in _MODULES:
            _LOG.info("[benchmark] Unknown module '%s', skipping.", name)
            continue

        model = _load_model(name, args.trained_dir)
        dummy = _MODULES[name]["dummy_fn"]()
        result = benchmark_module(
            model, dummy,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        all_results[name] = result

        module_path = out_dir / f"{name}_benchmark.json"
        module_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _LOG.info("[benchmark] %s: %.2fms  %.1f FPS  %.1f MB", name, result["avg_latency_ms"], result["fps"], result["peak_memory_mb"])

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    _LOG.info("\n[benchmark] Summary written to %s", summary_path)

    _LOG.info("\n%12s  %8s  %8s  %8s  %8s  %8s", "Module", "Avg(ms)", "P50(ms)", "P95(ms)", "FPS", "Mem(MB)")
    _LOG.info("-" * 62)
    for name, r in all_results.items():
        _LOG.info("%12s  %8.2f  %8.2f  %8.2f  %8.1f  %8.1f", name, r["avg_latency_ms"], r["p50_ms"], r["p95_ms"], r["fps"], r["peak_memory_mb"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
