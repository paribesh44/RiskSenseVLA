#!/usr/bin/env python3
"""Export lightweight model stubs to TorchScript and ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class ExportablePerceptionStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="artifacts/exports")
    p.add_argument("--onnx-opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = ExportablePerceptionStub().eval()
    dummy = torch.randn(1, 3, 224, 224)

    ts_path = out / "perception_stub.ts"
    traced = torch.jit.trace(model, dummy)
    traced.save(str(ts_path))

    onnx_path = out / "perception_stub.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=args.onnx_opset,
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print(f"Exported TorchScript: {ts_path}")
    print(f"Exported ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
