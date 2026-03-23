#!/usr/bin/env python3
"""Parse realtime JSONL logs into analysis-ready per-track rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from risksense_vla.io import load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log-jsonl", default="outputs/realtime_log.jsonl")
    p.add_argument("--output-csv", default="outputs/temporal_analysis.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.log_jsonl)
    rows: list[dict[str, object]] = []
    for rec in records:
        frame_id = int(rec.get("frame_id", -1))
        timestamp = float(rec.get("timestamp", 0.0))
        for item in rec.get("track_metrics", []) or []:
            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "track_id": item.get("track_id", ""),
                    "hazard_score": item.get("hazard_score", 0.0),
                    "predicted_action": item.get("predicted_action", ""),
                    "actual_action": item.get("actual_action", ""),
                    "object_lifetime": item.get("object_lifetime", 0),
                    "memory_strength": item.get("memory_strength", 0.0),
                    "hazard_history_len": len(item.get("hazard_history", [])),
                }
            )
        for item in rec.get("horizon_actuals", []) or []:
            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "track_id": item.get("track_id", ""),
                    "hazard_score": "",
                    "predicted_action": item.get("predicted_action", ""),
                    "actual_action": item.get("actual_action", ""),
                    "object_lifetime": "",
                    "memory_strength": "",
                    "hazard_history_len": "",
                    "source_frame_id": item.get("source_frame_id", frame_id),
                    "target_frame_id": item.get("target_frame_id", frame_id),
                    "horizon_seconds": item.get("horizon_seconds", ""),
                    "aligned": 1,
                }
            )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "frame_id",
        "timestamp",
        "track_id",
        "hazard_score",
        "predicted_action",
        "actual_action",
        "object_lifetime",
        "memory_strength",
        "hazard_history_len",
        "source_frame_id",
        "target_frame_id",
        "horizon_seconds",
        "aligned",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
