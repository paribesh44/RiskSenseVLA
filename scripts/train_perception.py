#!/usr/bin/env python3
"""Config-driven perception training with AMP, QAT, and validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from risksense_vla.config import load_config
from risksense_vla.train import ModuleTrainer, apply_qat, train_val_split

_LOG = logging.getLogger(__name__)


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
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="trained_models/perception/checkpoint.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)

    x = torch.randn(128, 3, 128, 128)
    y = torch.randn(128, 256)
    full_ds = TensorDataset(x, y)
    train_ds, val_ds = train_val_split(full_ds, val_ratio=args.val_split, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0) if val_ds else None

    model = TinyPerceptionModel()
    model = apply_qat(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    log_path = Path(args.output).parent / "train_log.jsonl"
    trainer = ModuleTrainer(model, opt, loss_fn, cfg, log_path=log_path)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    trainer.save_checkpoint(args.output)
    _LOG.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
