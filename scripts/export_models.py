#!/usr/bin/env python3
"""Export all trained VLA modules to TorchScript and ONNX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

_LOG = logging.getLogger(__name__)

from risksense_vla.config import load_config
from risksense_vla.train import convert_qat, export_module

from train_perception import TinyPerceptionModel
from train_hazard_vlm import TinyHazardNet
from risksense_vla.hoi import PredictiveHOINet


_MODULE_REGISTRY: dict[str, dict] = {
    "perception": {
        "model_cls": TinyPerceptionModel,
        "ckpt_default": "trained_models/perception/checkpoint.pt",
    },
    "hoi": {
        "model_cls": PredictiveHOINet,
        "ckpt_default": "trained_models/hoi/checkpoint.pt",
    },
    "hazard": {
        "model_cls": TinyHazardNet,
        "ckpt_default": "trained_models/hazard/checkpoint.pt",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default=None)
    p.add_argument("--trained-dir", default="trained_models")
    p.add_argument("--out-dir", default="exported_models")
    p.add_argument("--modules", nargs="+", default=["perception", "hoi", "hazard"])
    p.add_argument("--convert-qat", action="store_true", help="Run QAT conversion before export.")
    p.add_argument("--onnx-opset", type=int, default=17)
    return p.parse_args()


def _load_model(module_name: str, trained_dir: str) -> torch.nn.Module:
    reg = _MODULE_REGISTRY[module_name]
    ckpt_path = Path(trained_dir) / module_name / "checkpoint.pt"
    if not ckpt_path.exists():
        _LOG.info("[export] No checkpoint at %s, using fresh weights.", ckpt_path)
        return reg["model_cls"]()

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = reg["model_cls"]()
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=False)
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config) if args.backend_config else load_config(args.config)

    for name in args.modules:
        if name not in _MODULE_REGISTRY:
            _LOG.info("[export] Unknown module '%s', skipping.", name)
            continue

        model = _load_model(name, args.trained_dir)
        if args.convert_qat:
            try:
                model = convert_qat(model)
            except Exception as exc:
                _LOG.warning("[export] QAT conversion skipped for %s: %s", name, exc)

        model.eval()
        results = export_module(model, name, cfg, args.out_dir)
        for fmt, path in results.items():
            _LOG.info("[export] %s -> %s: %s", name, fmt, path)

        _verify_roundtrip(name, results)

    _LOG.info("[export] Done.")


def _verify_roundtrip(module_name: str, results: dict[str, Path]) -> None:
    ts_path = results.get("torchscript")
    if ts_path and ts_path.exists():
        try:
            loaded = torch.jit.load(str(ts_path))
            loaded.eval()
            _LOG.info("[export] %s TorchScript round-trip OK", module_name)
        except Exception as exc:
            _LOG.warning("[export] %s TorchScript round-trip FAILED: %s", module_name, exc)

    onnx_path = results.get("onnx")
    if onnx_path and onnx_path.exists():
        try:
            import onnxruntime as ort
            ort.InferenceSession(str(onnx_path))
            _LOG.info("[export] %s ONNX round-trip OK", module_name)
        except Exception as exc:
            _LOG.warning("[export] %s ONNX round-trip FAILED: %s", module_name, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
