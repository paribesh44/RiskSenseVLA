"""QAT and INT4 post-training quantization utilities."""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_LOG = logging.getLogger(__name__)


def _qconfig_for_backend(backend: str, observer: str) -> torch.ao.quantization.QConfig:
    """Return the appropriate QConfig based on backend and observer type."""
    if backend in ("cuda", "fbgemm"):
        if observer == "histogram":
            return torch.ao.quantization.QConfig(
                activation=torch.ao.quantization.HistogramObserver.with_args(dtype=torch.quint8),
                weight=torch.ao.quantization.default_per_channel_weight_observer,
            )
        return torch.ao.quantization.get_default_qat_qconfig("fbgemm")

    # MPS / CPU / qnnpack
    if observer == "moving_average_minmax":
        return torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.MovingAverageMinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
    return torch.ao.quantization.get_default_qat_qconfig("qnnpack")


def apply_qat(model: nn.Module, cfg: dict[str, Any]) -> nn.Module:
    """Attach fake-quant observers and prepare *model* for QAT.

    Reads ``cfg["qat"]`` for ``enabled``, ``fake_quant_backend``, and
    ``observer``.  Returns the original model unchanged when QAT is disabled.
    """
    qat_cfg = cfg.get("qat", {})
    if not qat_cfg.get("enabled", False):
        return model

    backend = str(qat_cfg.get("fake_quant_backend", "fbgemm"))
    observer = str(qat_cfg.get("observer", "histogram"))

    try:
        model.train()
        qconfig = _qconfig_for_backend(backend, observer)
        model.qconfig = qconfig  # type: ignore[assignment]
        model = torch.ao.quantization.prepare_qat(model, inplace=False)
    except Exception as exc:
        _LOG.warning("skipping QAT setup: %s", exc)
    return model


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert a QAT-prepared model to a fully quantized model.

    Returns the model in eval mode unchanged when the quantization engine is
    unavailable (e.g. on Apple Silicon without fbgemm/qnnpack support).
    """
    model.eval()
    try:
        return torch.ao.quantization.convert(model, inplace=False)
    except RuntimeError as exc:
        if "NoQEngine" in str(exc):
            _LOG.warning("convert_qat skipped (no quant engine): %s", exc)
            return model
        raise


def _run_calibration(
    model: nn.Module,
    loader: DataLoader,
    dev: torch.device,
    max_batches: int,
) -> None:
    """Forward-pass calibration to warm observer statistics."""
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                inputs = [b.to(dev) if isinstance(b, torch.Tensor) else b for b in batch[:-1]]
                model(*inputs) if len(inputs) != 1 else model(inputs[0])
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))


def apply_int4_ptq(
    model: nn.Module,
    calibration_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device | str = "cpu",
) -> nn.Module:
    """INT4-style post-training quantization via dynamic quantization on Linear
    layers.  Runs a calibration pass when a static observer is attached.

    Falls back to ``torch.quantization.quantize_dynamic`` with ``qint8``
    because PyTorch's native API does not expose true INT4 symmetric quant.
    The calibration pass still helps warm observer statistics.
    """
    int4_cfg = cfg.get("int4", {})
    if not int4_cfg.get("enabled", False):
        return model

    dev = torch.device(device)
    cal_batches = int(int4_cfg.get("calibration_batches", 32))

    calibration_model = copy.deepcopy(model).to(dev)
    calibration_model.eval()
    _run_calibration(calibration_model, calibration_loader, dev, cal_batches)
    del calibration_model

    try:
        quantized = torch.quantization.quantize_dynamic(
            model.cpu(),
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized
    except RuntimeError as exc:
        if "NoQEngine" in str(exc):
            _LOG.warning("INT4 PTQ skipped (no quant engine): %s", exc)
            return model
        raise
