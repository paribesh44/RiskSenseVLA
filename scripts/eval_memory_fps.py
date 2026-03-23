#!/usr/bin/env python3
"""Benchmark hazard-memory update latency and throughput."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from risksense_vla.memory import HazardAwareMemory
from risksense_vla.types import PerceptionDetection

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--bench-frames", type=int, default=300)
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--output-json", default="outputs/memory_fps.json")
    return parser.parse_args()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), p))


def _build_frame_batch(
    num_frames: int,
    max_objects: int,
    emb_dim: int,
    seed: int,
) -> list[tuple[list[PerceptionDetection], list[float]]]:
    rng = np.random.default_rng(seed)
    labels = ["person", "knife", "glass", "bottle", "vehicle", "stove"]
    batch: list[tuple[list[PerceptionDetection], list[float]]] = []
    for _ in range(max(1, num_frames)):
        obj_count = int(rng.integers(low=max(1, max_objects // 4), high=max_objects + 1))
        detections: list[PerceptionDetection] = []
        hazards: list[float] = []
        embeddings_np = rng.random((obj_count, emb_dim), dtype=np.float32)
        for det_idx in range(obj_count):
            x1 = int(rng.integers(0, 580))
            y1 = int(rng.integers(0, 340))
            w = int(rng.integers(24, 180))
            h = int(rng.integers(24, 180))
            mask = torch.zeros((360, 640), dtype=torch.float32)
            mask[y1 : y1 + h, x1 : x1 + w] = 1.0
            detections.append(
                PerceptionDetection(
                    track_id=f"t{int(rng.integers(0, max_objects * 2))}",
                    label=labels[int(rng.integers(0, len(labels)))],
                    confidence=float(rng.uniform(0.45, 0.99)),
                    bbox_xyxy=(x1, y1, x1 + w, y1 + h),
                    mask=mask,
                    clip_embedding=torch.from_numpy(embeddings_np[det_idx].copy()),
                )
            )
            hazards.append(float(rng.uniform(0.0, 1.0)))
        batch.append((detections, hazards))
    return batch


def _benchmark(memory: HazardAwareMemory, batch: list[tuple[list[PerceptionDetection], list[float]]]) -> dict[str, float]:
    latencies_ms: list[float] = []
    active_object_counts: list[float] = []
    timestamp = 0.0
    for detections, hazard_scores in batch:
        t0 = time.perf_counter()
        state = memory.update(
            timestamp=timestamp,
            detections=detections,
            hazards=hazard_scores,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        active_object_counts.append(float(len(state.objects)))
        timestamp += 1.0 / 25.0
    avg_latency = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
    fps = float(1000.0 / avg_latency) if avg_latency > 0 else 0.0
    return {
        "frames": float(len(batch)),
        "avg_active_objects": float(sum(active_object_counts) / len(active_object_counts))
        if active_object_counts
        else 0.0,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": _percentile(latencies_ms, 50),
        "p95_latency_ms": _percentile(latencies_ms, 95),
        "max_latency_ms": max(latencies_ms) if latencies_ms else 0.0,
        "avg_fps": fps,
    }


def main() -> None:
    args = parse_args()
    memory = HazardAwareMemory(emb_dim=args.emb_dim)

    warmup_batch = _build_frame_batch(
        num_frames=max(0, args.warmup_frames),
        max_objects=max(1, args.max_objects),
        emb_dim=args.emb_dim,
        seed=11,
    )
    bench_batch = _build_frame_batch(
        num_frames=max(1, args.bench_frames),
        max_objects=max(1, args.max_objects),
        emb_dim=args.emb_dim,
        seed=29,
    )

    if warmup_batch:
        _benchmark(memory, warmup_batch)
    metrics = _benchmark(memory, bench_batch)
    report = {
        "bench_frames": int(metrics["frames"]),
        "warmup_frames": max(0, args.warmup_frames),
        "max_objects": max(1, args.max_objects),
        "embedding_dim": args.emb_dim,
        "metrics": metrics,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _LOG.info("%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
