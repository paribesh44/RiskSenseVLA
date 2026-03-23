"""Shared training infrastructure: loop, AMP, checkpointing, JSONL logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from risksense_vla.runtime import pick_backend

_LOG = logging.getLogger(__name__)


def train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """Split *dataset* into train / val subsets.  Returns ``(train, None)`` when
    val_ratio is non-positive or the dataset is too small to split."""
    n = len(dataset)  # type: ignore[arg-type]
    if n < 2 or val_ratio <= 0.0:
        return dataset, None
    val_n = max(1, int(round(n * val_ratio)))
    train_n = n - val_n
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_n, val_n], generator=gen)


class ModuleTrainer:
    """Config-driven trainer with AMP, gradient clipping, validation, and JSONL
    logging.  Works with any ``nn.Module`` / loss / optimizer combination."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module | Callable[..., torch.Tensor],
        cfg: dict[str, Any],
        *,
        metrics_fn: Callable[[nn.Module, DataLoader, torch.device], dict[str, float]] | None = None,
        log_path: str | Path | None = None,
        grad_clip_norm: float = 1.0,
    ) -> None:
        self.cfg = cfg
        backend = pick_backend(cfg.get("runtime", {}).get("backend", "cpu"))
        self.device = torch.device(backend.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.grad_clip_norm = grad_clip_norm

        rt = cfg.get("runtime", {})
        self.use_amp = bool(rt.get("mixed_precision", False)) and self.device.type not in ("cpu",)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        self._log_path = Path(log_path) if log_path else None
        self._epoch: int = 0
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
    ) -> list[dict[str, Any]]:
        """Run *epochs* training epochs and return per-epoch history."""
        for _ in range(epochs):
            t0 = time.perf_counter()
            train_loss = self._train_one_epoch(train_loader)
            val_result = self.validate(val_loader) if val_loader is not None else {}
            elapsed = time.perf_counter() - t0
            record: dict[str, Any] = {
                "epoch": self._epoch,
                "train_loss": train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": round(elapsed, 3),
                **{f"val_{k}": v for k, v in val_result.items()},
            }
            record["peak_memory_mb"] = self._get_peak_memory_mb()
            self._history.append(record)
            self._log(record)
            self._epoch += 1
        return self._history

    def _get_peak_memory_mb(self) -> float:
        """Return peak memory usage in MB for the current device."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            return rusage.ru_maxrss / 1024  # macOS reports in bytes
        except (ImportError, AttributeError):
            return 0.0

    def validate(self, val_loader: DataLoader | None) -> dict[str, float]:
        if val_loader is None:
            return {}

        if self.metrics_fn is not None:
            return self.metrics_fn(self.model, val_loader, self.device)

        self.model.eval()
        total_loss = 0.0
        steps = 0
        with torch.inference_mode():
            for batch in val_loader:
                inputs, targets = self._unpack_batch(batch)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.float16):
                    preds = self.model(*inputs) if isinstance(inputs, (list, tuple)) else self.model(inputs)
                    loss = self.loss_fn(preds, targets)
                total_loss += float(loss.item())
                steps += 1
        return {"loss": total_loss / max(steps, 1)}

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self._epoch,
            "history": self._history,
            "config": self.cfg,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, out)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load model and optimizer state from a checkpoint file."""
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            _LOG.warning("weights_only=True failed for %s; falling back to weights_only=False", path)
            payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"], strict=False)
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self._epoch = int(payload.get("epoch", 0))
        self._history = payload.get("history", [])
        return payload

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.float16):
                preds = self.model(*inputs) if isinstance(inputs, (list, tuple)) else self.model(inputs)
                loss = self.loss_fn(preds, targets)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += float(loss.item())
            steps += 1
        return total_loss / max(steps, 1)

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        """Default unpacking: batch is a tuple/list of ``(inputs, targets)``
        where inputs go to device.  Override for custom layouts."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            x = x.to(self.device) if isinstance(x, torch.Tensor) else x
            y = y.to(self.device) if isinstance(y, torch.Tensor) else y
            return x, y
        if isinstance(batch, (list, tuple)):
            return (
                [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch[:-1]],
                batch[-1].to(self.device) if isinstance(batch[-1], torch.Tensor) else batch[-1],
            )
        raise TypeError(f"Cannot unpack batch of type {type(batch).__name__}")

    def _log(self, record: dict[str, Any]) -> None:
        msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
        _LOG.info(msg)
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
