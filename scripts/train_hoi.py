#!/usr/bin/env python3
"""Config-driven zero-shot HOI and predictive embedding training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hapvla.config import load_config
from hapvla.runtime import pick_backend


class TinyHOINet(nn.Module):
    def __init__(self, emb_dim: int = 256, num_actions: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(emb_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.current_head = nn.Linear(256, num_actions)
        self.future_head = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.current_head(z), self.future_head(z)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default="configs/backend_mps.yaml")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="artifacts/hoi.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    device = torch.device(backend.device)

    x = torch.randn(2048, 256)
    y_cur = torch.randint(0, 12, (2048,))
    y_fut = torch.randint(0, 12, (2048,))
    loader = DataLoader(TensorDataset(x, y_cur, y_fut), batch_size=args.batch_size, shuffle=True)

    model = TinyHOINet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    use_amp = bool(cfg.get("runtime", {}).get("mixed_precision", True)) and backend.device != "cpu"

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb_cur, yb_fut in loader:
            xb = xb.to(device)
            yb_cur = yb_cur.to(device)
            yb_fut = yb_fut.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                cur_logits, fut_logits = model(xb)
                # Predictive HOI objective uses both present and short-horizon future labels.
                loss = 0.6 * ce(cur_logits, yb_cur) + 0.4 * ce(fut_logits, yb_fut)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"[hoi] epoch={epoch} loss={total/len(loader):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
