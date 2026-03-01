#!/usr/bin/env python3
"""Config-driven hazard reasoner fine-tuning skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from risksense_vla.config import load_config
from risksense_vla.runtime import pick_backend


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
    p.add_argument("--output", default="artifacts/hazard_reasoner.pt")
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
        # Stand-in embedding; replace with textual+visual encoder in full model.
        vecs.append(torch.randn(256))
    return torch.stack(vecs), torch.tensor(labels, dtype=torch.long)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    device = torch.device(backend.device)
    x, y = load_or_mock_data(args.synthetic_jsonl)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    model = TinyHazardNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    use_amp = bool(cfg.get("runtime", {}).get("mixed_precision", True)) and backend.device != "cpu"

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                logits = model(xb)
                loss = ce(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"[hazard] epoch={epoch} loss={total/len(loader):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
