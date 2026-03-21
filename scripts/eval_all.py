#!/usr/bin/env python3
"""Aggregate all evaluation metrics and generate diagnostics."""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

from risksense_vla.eval import aggregate_sequences, evaluate_sequence, plot_failure_heatmap, plot_hoi_trajectory
from risksense_vla.io import load_jsonl

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log-jsonl", default="outputs/realtime_log.jsonl")
    p.add_argument("--report-json", default="outputs/eval_report.json")
    p.add_argument("--plots-dir", default="outputs/plots")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    recs = load_jsonl(args.log_jsonl)
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
            "HazardLeadTimeMean": seq.hazard_lead_time_mean,
            "HazardLeadTimeMedian": seq.hazard_lead_time_median,
            "prediction_accuracy@1s": seq.prediction_accuracy_by_horizon.get(1, 0.0),
            "prediction_accuracy@2s": seq.prediction_accuracy_by_horizon.get(2, 0.0),
            "prediction_accuracy@3s": seq.prediction_accuracy_by_horizon.get(3, 0.0),
        },
        "aggregate": agg,
    }
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_dir = Path(args.plots_dir)
    plot_failure_heatmap(recs, str(plot_dir / "hazard_attention_heatmap.png"))
    plot_hoi_trajectory(recs, str(plot_dir / "hoi_trajectory.png"))
    _LOG.info("%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
