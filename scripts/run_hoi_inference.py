#!/usr/bin/env python3
"""Run predictive HOI inference and log per-frame current/future outputs."""

from __future__ import annotations

import argparse
import logging
import json
import time
from pathlib import Path

from risksense_vla.config import load_config
from risksense_vla.hoi import PredictiveHOIModule
from risksense_vla.io import VideoInput, resolve_source
from risksense_vla.memory import HazardAwareMemory
from risksense_vla.perception import OpenVocabPerception
from risksense_vla.runtime import pick_backend

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default=None)
    p.add_argument("--source", default=None, help="Camera index or video file path.")
    p.add_argument("--labels", default="", help="Comma-separated open-vocab labels.")
    p.add_argument("--max-frames", type=int, default=120, help="0 means run until stream ends.")
    p.add_argument("--log-jsonl", default="outputs/hoi_inference.jsonl")
    p.add_argument("--checkpoint", default=None, help="Optional fine-tuned HOI checkpoint.")
    p.add_argument("--horizon-seconds", type=int, default=None)
    return p.parse_args()


def _parse_labels(raw: str) -> list[str] | None:
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    return labels if labels else None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    _LOG.info("[hoi-infer] backend=%s device=%s", backend.name, backend.device)

    src = args.source if args.source is not None else str(cfg.get("io", {}).get("source", 0))
    width = int(cfg.get("io", {}).get("width", 1280))
    height = int(cfg.get("io", {}).get("height", 720))
    fps = int(cfg.get("runtime", {}).get("target_fps", 25))
    horizon = int(args.horizon_seconds or cfg.get("hazard", {}).get("future_horizon_seconds", 3))
    horizon = max(1, min(3, horizon))
    capture = VideoInput(resolve_source(src), width=width, height=height, target_fps=fps)
    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
    memory = HazardAwareMemory(emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)))
    hoi = PredictiveHOIModule(
        future_horizon_seconds=horizon,
        emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
        checkpoint_path=args.checkpoint,
    )

    out_path = Path(args.log_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = _parse_labels(args.labels)
    latencies: list[float] = []
    with out_path.open("w", encoding="utf-8") as fh:
        try:
            for captured in capture.stream():
                t0 = time.perf_counter()
                detections = perception.infer(captured.bgr, labels=labels)
                t1 = time.perf_counter()
                mem = memory.update(
                    timestamp=captured.timestamp,
                    detections=detections,
                    hazards=None,
                )
                t2 = time.perf_counter()
                out = hoi.infer(
                    memory_state=mem,
                    object_detections=detections,
                    timestamp=captured.timestamp,
                    horizon_seconds=horizon,
                )
                t3 = time.perf_counter()

                total_ms = (t3 - t0) * 1000.0
                latencies.append(total_ms)
                record = {
                    "frame_id": captured.frame_index,
                    "timestamp": captured.timestamp,
                    "detections": [
                        {
                            "track_id": d.track_id,
                            "label": d.label,
                            "confidence": float(d.confidence),
                            "bbox_xyxy": list(d.bbox_xyxy),
                        }
                        for d in detections
                    ],
                    "hoi_current": [
                        {
                            "subject": h.subject,
                            "action": h.action,
                            "object": h.object,
                            "confidence": float(h.confidence),
                            "t_start": float(h.t_start),
                            "t_end": float(h.t_end),
                        }
                        for h in out.hoi_current
                    ],
                    "hoi_future_action_labels": out.future_action_labels,
                    "hoi_future_embeddings": out.hoi_future_embeddings.detach().cpu().to(dtype=out.hoi_future_embeddings.dtype).tolist(),
                    "latency_ms": {
                        "perception": (t1 - t0) * 1000.0,
                        "memory": (t2 - t1) * 1000.0,
                        "hoi": (t3 - t2) * 1000.0,
                        "total": total_ms,
                    },
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()

                if captured.frame_index % 20 == 0:
                    _LOG.info(
                        "[hoi-infer] frame=%s dets=%s hoi=%s lat=%.1fms",
                        captured.frame_index,
                        len(detections),
                        len(out.hoi_current),
                        total_ms,
                    )
                if args.max_frames and captured.frame_index + 1 >= args.max_frames:
                    break
        finally:
            capture.close()

    if latencies:
        avg_ms = sum(latencies) / len(latencies)
        fps_now = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
        _LOG.info("[hoi-infer] frames=%s avg_latency=%.2fms fps=%.2f", len(latencies), avg_ms, fps_now)
    _LOG.info("[hoi-infer] wrote %s", out_path)


if __name__ == "__main__":
    main()
