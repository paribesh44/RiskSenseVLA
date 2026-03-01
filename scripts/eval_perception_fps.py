#!/usr/bin/env python3
"""Benchmark perception inference throughput and latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hapvla.config import load_config
from hapvla.io import VideoInput, resolve_source
from hapvla.perception import OpenVocabPerception
from hapvla.runtime import pick_backend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default=None)
    p.add_argument("--source", default=None, help="Camera index or video path for real capture benchmark.")
    p.add_argument("--mode", choices=["synthetic", "source"], default="synthetic")
    p.add_argument("--labels", default="", help="Comma-separated open-vocab labels.")
    p.add_argument("--warmup-frames", type=int, default=10)
    p.add_argument("--bench-frames", type=int, default=120)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--output-json", default="outputs/perception_fps.json")
    return p.parse_args()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), p))


def _benchmark_on_frames(
    perception: OpenVocabPerception, frames: list[np.ndarray], labels: list[str] | None
) -> dict[str, float]:
    latencies_ms: list[float] = []
    total_detections = 0
    for frame in frames:
        t0 = time.perf_counter()
        out = perception.infer(frame, labels=labels)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        total_detections += len(out.detections)
    avg_latency = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
    fps = float(1000.0 / avg_latency) if avg_latency > 0 else 0.0
    return {
        "frames": float(len(frames)),
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": _percentile(latencies_ms, 50),
        "p95_latency_ms": _percentile(latencies_ms, 95),
        "avg_fps": fps,
        "avg_detections": float(total_detections / len(frames)) if frames else 0.0,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()] if args.labels else None

    warmup_frames = max(0, args.warmup_frames)
    bench_frames = max(1, args.bench_frames)

    if args.mode == "synthetic":
        rng = np.random.default_rng(42)
        warmup = [rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8) for _ in range(warmup_frames)]
        bench = [rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8) for _ in range(bench_frames)]
    else:
        src = args.source if args.source is not None else str(cfg.get("io", {}).get("source", 0))
        cap = VideoInput(resolve_source(src), width=args.width, height=args.height, target_fps=60)
        warmup = []
        bench = []
        try:
            for captured in cap.stream():
                if len(warmup) < warmup_frames:
                    warmup.append(captured.bgr.copy())
                elif len(bench) < bench_frames:
                    bench.append(captured.bgr.copy())
                else:
                    break
        finally:
            cap.close()
        if len(bench) < bench_frames:
            raise RuntimeError(
                f"Could not collect enough source frames for benchmark ({len(bench)}/{bench_frames})."
            )

    if warmup:
        _benchmark_on_frames(perception, warmup, labels)
    metrics = _benchmark_on_frames(perception, bench, labels)
    report = {
        "mode": args.mode,
        "backend": backend.name,
        "device": backend.device,
        "frame_size": {"width": args.width, "height": args.height},
        "warmup_frames": warmup_frames,
        "bench_frames": bench_frames,
        "metrics": metrics,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
