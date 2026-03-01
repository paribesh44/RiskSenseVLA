#!/usr/bin/env python3
"""Config-driven perception training skeleton with AMP + QAT hooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hapvla.config import load_config
from hapvla.runtime import pick_backend


class TinyPerceptionModel(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.head(feat)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default="configs/backend_mps.yaml")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default="artifacts/perception.pt")
    return p.parse_args()


def maybe_prepare_qat(model: nn.Module, cfg: dict) -> nn.Module:
    if bool(cfg.get("qat", {}).get("enabled", False)):
        try:
            model.train()
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
            return torch.ao.quantization.prepare_qat(model)
        except Exception as exc:  # pragma: no cover - environment dependent.
            print(f"[perception] skipping QAT setup: {exc}")
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    device = torch.device(backend.device)

    x = torch.randn(128, 3, 128, 128)
    y = torch.randn(128, 256)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    model = TinyPerceptionModel().to(device)
    model = maybe_prepare_qat(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    use_amp = bool(cfg.get("runtime", {}).get("mixed_precision", True)) and backend.device != "cpu"
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.item())
        print(f"[perception] epoch={epoch} loss={running/len(loader):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
