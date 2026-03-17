#!/usr/bin/env python3
"""Benchmark strict Phase-4 runtime latency and FPS."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

from risksense_vla.config import load_config
from risksense_vla.hazard import DistilledHazardReasoner
from risksense_vla.hoi import PredictiveHOIModule
from risksense_vla.io import VideoInput, resolve_source
from risksense_vla.memory import HazardAwareMemory
from risksense_vla.perception import OpenVocabPerception
from risksense_vla.runtime import pick_backend

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend-config", default=None)
    parser.add_argument("--source", default=None, help="Camera index or video file path.")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--min-fps", type=float, default=10.0)
    parser.add_argument("--require-gpu", action="store_true", default=True)
    parser.add_argument("--output-json", default="outputs/phase4_benchmark.json")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    if args.require_gpu and backend.name not in {"mps", "cuda"}:
        raise RuntimeError("Phase-4 benchmark requires GPU backend (mps/cuda).")

    source = args.source if args.source is not None else str(cfg.get("io", {}).get("source", 0))
    width = int(cfg.get("io", {}).get("width", 1280))
    height = int(cfg.get("io", {}).get("height", 720))
    fps = int(cfg.get("runtime", {}).get("target_fps", 25))
    capture = VideoInput(resolve_source(source), width=width, height=height, target_fps=fps)

    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
    memory = HazardAwareMemory(emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)))
    hoi = PredictiveHOIModule(
        future_horizon_seconds=int(cfg.get("hazard", {}).get("future_horizon_seconds", 3)),
        emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
    )
    reasoner = DistilledHazardReasoner(
        alert_threshold=float(cfg.get("hazard", {}).get("alert_threshold", 0.65)),
        checkpoint_path=str(cfg.get("hazard", {}).get("reasoner_checkpoint", "artifacts/hazard_reasoner.pt")),
        fallback_mode=str(cfg.get("hazard", {}).get("reasoner_fallback_mode", "blend")),
        emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
        backend_type=str(cfg.get("hazard", {}).get("backend_type", "phi4_mm")),
        max_tokens=int(cfg.get("hazard", {}).get("max_tokens", 64)),
        temperature=float(cfg.get("hazard", {}).get("temperature", 0.2)),
        quantized=bool(cfg.get("hazard", {}).get("quantized", False)),
        lightweight_mode=bool(cfg.get("hazard", {}).get("lightweight_mode", False)),
        phi4_model_id=str(cfg.get("hazard", {}).get("phi4_model_id", "microsoft/Phi-4-multimodal-instruct")),
        phi4_precision=str(cfg.get("hazard", {}).get("phi4_precision", "int8")),
        phi4_estimated_vram_gb=float(cfg.get("hazard", {}).get("phi4_estimated_vram_gb", 10.0)),
        explain=bool(cfg.get("hazard", {}).get("explain", True)),
        debug_prompt=bool(cfg.get("hazard", {}).get("debug_prompt", False)),
    )

    per_ms: list[float] = []
    mem_ms: list[float] = []
    hoi_ms: list[float] = []
    hazard_ms: list[float] = []
    e2e_ms: list[float] = []
    frame_count = 0

    try:
        for captured in capture.stream():
            if frame_count >= max(1, args.frames):
                break
            t0 = time.perf_counter()
            detections = perception.infer(captured.bgr)
            t1 = time.perf_counter()
            mem = memory.update(timestamp=captured.timestamp, detections=detections, hazards=None)
            t2 = time.perf_counter()
            hoi_out = hoi.infer(memory_state=mem, object_detections=detections, timestamp=captured.timestamp)
            t3 = time.perf_counter()
            hazard_out = reasoner.predict_hazard(
                hoi_current=hoi_out.hoi_current,
                hoi_future_embeddings=hoi_out.hoi_future_embeddings,
                memory_state=mem,
                frame_bgr=captured.bgr,
            )
            t4 = time.perf_counter()
            memory.update(
                timestamp=captured.timestamp,
                detections=detections,
                hazards=None,
                hazard_events=hazard_out.hazards,
            )
            t5 = time.perf_counter()

            per_ms.append((t1 - t0) * 1000.0)
            mem_ms.append((t2 - t1 + t5 - t4) * 1000.0)
            hoi_ms.append((t3 - t2) * 1000.0)
            hazard_ms.append((t4 - t3) * 1000.0)
            e2e_ms.append((t5 - t0) * 1000.0)
            frame_count += 1
    finally:
        capture.close()

    if frame_count < max(1, args.frames):
        raise RuntimeError(f"Could only benchmark {frame_count}/{args.frames} frames from source '{source}'.")

    avg_e2e = _mean(e2e_ms)
    end_to_end_fps = (1000.0 / avg_e2e) if avg_e2e > 0 else 0.0
    report = {
        "backend": backend.name,
        "device": backend.device,
        "frames": frame_count,
        "perception_ms": _mean(per_ms),
        "memory_ms": _mean(mem_ms),
        "hoi_ms": _mean(hoi_ms),
        "hazard_ms": _mean(hazard_ms),
        "end_to_end_fps": float(end_to_end_fps),
        "min_fps_threshold": float(args.min_fps),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _LOG.info("%s", json.dumps(report, indent=2))

    if end_to_end_fps < float(args.min_fps):
        raise SystemExit(
            f"Phase-4 benchmark failed: end_to_end_fps={end_to_end_fps:.2f} < threshold={args.min_fps:.2f}"
        )


if __name__ == "__main__":
    main()
