"""TorchScript and ONNX model export utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

_LOG = logging.getLogger(__name__)


def _ensure_cpu_for_export(
    model: nn.Module,
    dummy_input: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[nn.Module, torch.Tensor | tuple[torch.Tensor, ...]]:
    """Move model and inputs to CPU for export (avoids device mismatch)."""
    model = model.cpu()
    if isinstance(dummy_input, tuple):
        dummy_input = tuple(d.cpu() if isinstance(d, torch.Tensor) else d for d in dummy_input)
    elif isinstance(dummy_input, torch.Tensor):
        dummy_input = dummy_input.cpu()
    return model, dummy_input


def export_to_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor | tuple[torch.Tensor, ...],
    path: str | Path,
) -> Path:
    """Trace *model* with *dummy_input* and save as TorchScript."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model, dummy_input = _ensure_cpu_for_export(model, dummy_input)
    model.eval()
    if isinstance(dummy_input, tuple):
        traced = torch.jit.trace(model, dummy_input)
    else:
        traced = torch.jit.trace(model, (dummy_input,))
    traced.save(str(out))
    _LOG.info("TorchScript exported to %s", out)
    return out


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor | tuple[torch.Tensor, ...],
    path: str | Path,
    *,
    input_names: Sequence[str] = ("input",),
    output_names: Sequence[str] = ("output",),
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset: int = 17,
) -> Path:
    """Export *model* to ONNX format with optional dynamic axes."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model, dummy_input = _ensure_cpu_for_export(model, dummy_input)
    model.eval()

    if dynamic_axes is None:
        dynamic_axes = {name: {0: "batch"} for name in list(input_names) + list(output_names)}

    torch.onnx.export(
        model,
        dummy_input if isinstance(dummy_input, tuple) else (dummy_input,),
        str(out),
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    return out


# Module-specific dummy inputs
_DUMMY_SPECS: dict[str, dict[str, Any]] = {
    "perception": {
        "dummy_fn": lambda: torch.randn(1, 3, 128, 128),
        "input_names": ["image"],
        "output_names": ["embedding"],
    },
    "hoi": {
        "dummy_fn": lambda: (torch.randn(1, 256), torch.randn(1, 256)),
        "input_names": ["object_emb", "memory_emb"],
        "output_names": ["current_logits", "future_action_logits", "future_embedding"],
    },
    "hazard": {
        "dummy_fn": lambda: torch.randn(1, 256),
        "input_names": ["features"],
        "output_names": ["logits"],
    },
}


def export_module(
    model: nn.Module,
    module_name: str,
    cfg: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Path]:
    """High-level export: produces all formats listed in
    ``cfg["optimization"]["export_formats"]``."""
    out_dir = Path(out_dir) / module_name
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = cfg.get("optimization", {}).get("export_formats", ["torchscript", "onnx"])
    spec = _DUMMY_SPECS.get(module_name, _DUMMY_SPECS["hazard"])
    dummy = spec["dummy_fn"]()
    results: dict[str, Path] = {}

    if "torchscript" in formats:
        ts_path = export_to_torchscript(model, dummy, out_dir / "model.ts")
        results["torchscript"] = ts_path

    if "onnx" in formats:
        onnx_path = export_to_onnx(
            model,
            dummy,
            out_dir / "model.onnx",
            input_names=spec["input_names"],
            output_names=spec["output_names"],
        )
        results["onnx"] = onnx_path

    return results
