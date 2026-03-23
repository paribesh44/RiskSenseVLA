#!/usr/bin/env python3
"""Generate paper-ready CSV/Markdown tables from benchmark and ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from risksense_vla.experimental import method_display_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-json", default="outputs/ablations/ablation_results.json")
    parser.add_argument("--benchmark-json", default="outputs/phase4_benchmark.json")
    parser.add_argument("--output-dir", default="outputs/paper_tables")
    parser.add_argument("--write-examples", action="store_true")
    return parser.parse_args()


def _read_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _to_markdown(headers: list[str], rows: list[dict[str, Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body])


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def _normalize_method(row: dict[str, Any]) -> str:
    if "method_name" in row and row["method_name"]:
        return str(row["method_name"])
    memory_mode = str(row.get("memory_mode", ""))
    hoi_mode = str(row.get("hoi_mode", ""))
    if memory_mode == "naive":
        return method_display_name("naive")
    if hoi_mode == "frame_only":
        return method_display_name("frame_only")
    return method_display_name("hazard_aware")


def _load_ablation_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [r for r in data["results"] if isinstance(r, dict)]
    return []


def _load_benchmark_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [r for r in data["results"] if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abl_raw = _read_json(args.ablation_json)
    bench_raw = _read_json(args.benchmark_json)

    ablations = _load_ablation_rows(abl_raw)
    benchmarks = _load_benchmark_rows(bench_raw)

    table1_headers = ["Method", "Prediction@1s", "Prediction@2s", "Prediction@3s"]
    table1_rows = [
        {
            "Method": _normalize_method(row),
            "Prediction@1s": row.get("prediction_accuracy@1s", 0.0),
            "Prediction@2s": row.get("prediction_accuracy@2s", 0.0),
            "Prediction@3s": row.get("prediction_accuracy@3s", 0.0),
        }
        for row in ablations
    ]

    table2_headers = ["Method", "Hazard Lead Time", "THC", "FPS"]
    table2_rows = [
        {
            "Method": _normalize_method(row),
            "Hazard Lead Time": row.get("HazardLeadTimeMean", 0.0),
            "THC": row.get("THC", 0.0),
            "FPS": row.get("FPS", row.get("end_to_end_fps", 0.0)),
        }
        for row in ablations
    ]

    if not table2_rows and benchmarks:
        table2_rows = [
            {
                "Method": str(row.get("metadata", {}).get("method_name", method_display_name("hazard_aware"))),
                "Hazard Lead Time": row.get("hazard_lead_time_mean", 0.0),
                "THC": row.get("THC", 0.0),
                "FPS": row.get("end_to_end_fps", 0.0),
            }
            for row in benchmarks
        ]

    table3_headers = ["Occlusion", "Method", "Prediction@1s", "Hazard Lead Time"]
    table3_rows = [
        {
            "Occlusion": row.get("occlusion", 0.0),
            "Method": str(row.get("metadata", {}).get("method_name", method_display_name("hazard_aware"))),
            "Prediction@1s": row.get("prediction_accuracy@1s", 0.0),
            "Hazard Lead Time": row.get("hazard_lead_time_mean", 0.0),
        }
        for row in benchmarks
    ]

    _write_csv(out_dir / "table1_prediction.csv", table1_headers, table1_rows)
    _write_csv(out_dir / "table2_hazard_latency.csv", table2_headers, table2_rows)
    _write_csv(out_dir / "table3_occlusion.csv", table3_headers, table3_rows)

    (out_dir / "table1_prediction.md").write_text(
        _to_markdown(table1_headers, table1_rows),
        encoding="utf-8",
    )
    (out_dir / "table2_hazard_latency.md").write_text(
        _to_markdown(table2_headers, table2_rows),
        encoding="utf-8",
    )
    (out_dir / "table3_occlusion.md").write_text(
        _to_markdown(table3_headers, table3_rows),
        encoding="utf-8",
    )

    if args.write_examples:
        (out_dir / "example_occlusion_sweep.json").write_text(
            json.dumps(benchmarks, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
