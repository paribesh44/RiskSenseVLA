#!/usr/bin/env python3
"""Ablation experiment scaffold for memory, HOI, attention, and quantization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-csv", default="outputs/ablations.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        {
            "ablation": "hazard_memory_vs_naive",
            "memory_mode": "hazard_aware",
            "predictive_hoi": True,
            "semantic_attention": True,
            "quant_bits": 8,
            "fps": "",
            "thc": "",
            "haa": "",
            "rme": "",
        },
        {
            "ablation": "hazard_memory_vs_naive",
            "memory_mode": "naive",
            "predictive_hoi": True,
            "semantic_attention": True,
            "quant_bits": 8,
            "fps": "",
            "thc": "",
            "haa": "",
            "rme": "",
        },
        {
            "ablation": "predictive_vs_frame_hoi",
            "memory_mode": "hazard_aware",
            "predictive_hoi": False,
            "semantic_attention": True,
            "quant_bits": 8,
            "fps": "",
            "thc": "",
            "haa": "",
            "rme": "",
        },
        {
            "ablation": "semantic_attention_vs_uniform",
            "memory_mode": "hazard_aware",
            "predictive_hoi": True,
            "semantic_attention": False,
            "quant_bits": 8,
            "fps": "",
            "thc": "",
            "haa": "",
            "rme": "",
        },
        {
            "ablation": "int4_masking_impact",
            "memory_mode": "hazard_aware",
            "predictive_hoi": True,
            "semantic_attention": True,
            "quant_bits": 4,
            "fps": "",
            "thc": "",
            "haa": "",
            "rme": "",
        },
    ]

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote ablation template to {out}")


if __name__ == "__main__":
    main()
