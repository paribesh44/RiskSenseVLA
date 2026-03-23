#!/usr/bin/env python3
"""Config-driven hazard reasoner fine-tuning with QAT support and validation."""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from risksense_vla.config import load_config
from risksense_vla.train import ModuleTrainer, apply_qat, train_val_split

_LOG = logging.getLogger(__name__)


SEVERITY_TO_ID = {"low": 0, "medium": 1, "high": 2}


class TinyHazardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default="configs/backend_mps.yaml")
    p.add_argument("--synthetic-jsonl", default="data/synthetic/hazards.jsonl")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="trained_models/hazard/checkpoint.pt")
    return p.parse_args()


def load_or_mock_data(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    p = Path(path)
    if not p.exists():
        x = torch.randn(1024, 256)
        y = torch.randint(0, 3, (1024,))
        return x, y

    vecs, labels = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sev = rec.get("hazard_severity", "low")
        labels.append(SEVERITY_TO_ID.get(sev, 0))
        vecs.append(torch.randn(256))
    return torch.stack(vecs), torch.tensor(labels, dtype=torch.long)


def _hazard_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    steps = 0
    ce = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += float(ce(logits, yb).item())
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == yb).sum().item())
            total += int(yb.numel())
            steps += 1
    return {
        "loss": total_loss / max(steps, 1),
        "accuracy": correct / max(total, 1),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)

    x, y = load_or_mock_data(args.synthetic_jsonl)
    full_ds = TensorDataset(x, y)
    train_ds, val_ds = train_val_split(full_ds, val_ratio=args.val_split, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0) if val_ds else None

    model = TinyHazardNet()
    model = apply_qat(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    log_path = Path(args.output).parent / "train_log.jsonl"
    trainer = ModuleTrainer(
        model, opt, loss_fn, cfg,
        metrics_fn=_hazard_metrics,
        log_path=log_path,
    )
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    trainer.save_checkpoint(args.output)
    _LOG.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
