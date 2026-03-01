#!/usr/bin/env python3
"""Aggregate all evaluation metrics and generate diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hapvla.eval import aggregate_sequences, evaluate_sequence, plot_failure_heatmap, plot_hoi_trajectory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log-jsonl", default="outputs/realtime_log.jsonl")
    p.add_argument("--report-json", default="outputs/eval_report.json")
    p.add_argument("--plots-dir", default="outputs/plots")
    return p.parse_args()


def load_records(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def main() -> None:
    args = parse_args()
    recs = load_records(args.log_jsonl)
    seq = evaluate_sequence(recs)
    agg = aggregate_sequences([seq])
    report = {
        "per_sequence": {
            "THC": seq.thc,
            "HAA": seq.haa,
            "RME": seq.rme,
            "FPS": seq.fps,
            "LatencyMS": seq.latency_ms,
            "mAP": seq.detection_map,
        },
        "aggregate": agg,
    }
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_dir = Path(args.plots_dir)
    plot_failure_heatmap(recs, str(plot_dir / "hazard_attention_heatmap.png"))
    plot_hoi_trajectory(recs, str(plot_dir / "hoi_trajectory.png"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
