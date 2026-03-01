#!/usr/bin/env python3
"""End-to-end real-time inference loop for RiskSense-VLA."""

from __future__ import annotations

import argparse
import time

import cv2
import torch

from risksense_vla.attention import SemanticAttentionScheduler
from risksense_vla.config import load_config
from risksense_vla.hazard import LaCHazardReasoner
from risksense_vla.hoi import ProtoHOIPredictor
from risksense_vla.io import VideoInput, resolve_source
from risksense_vla.memory import HazardAwareMemory
from risksense_vla.perception import OpenVocabPerception
from risksense_vla.runtime import pick_backend
from risksense_vla.types import FrameData
from risksense_vla.viz import JsonlRunLogger, render_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default=None)
    p.add_argument("--source", default=None, help="Camera index or video file path.")
    p.add_argument("--max-frames", type=int, default=0, help="0 means run until stream ends.")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    print(f"[RiskSense-VLA] backend={backend.name} device={backend.device}")

    src = args.source if args.source is not None else str(cfg.get("io", {}).get("source", 0))
    width = int(cfg.get("io", {}).get("width", 1280))
    height = int(cfg.get("io", {}).get("height", 720))
    fps = int(cfg.get("runtime", {}).get("target_fps", 25))
    capture = VideoInput(resolve_source(src), width=width, height=height, target_fps=fps)

    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
    memory = HazardAwareMemory()
    hoi = ProtoHOIPredictor(future_horizon_seconds=int(cfg.get("hazard", {}).get("future_horizon_seconds", 3)))
    reasoner = LaCHazardReasoner(alert_threshold=float(cfg.get("hazard", {}).get("alert_threshold", 0.65)))
    attention = SemanticAttentionScheduler(
        threshold=float(cfg.get("attention", {}).get("semantic_attention_threshold", 0.6)),
        low_risk_scale=float(cfg.get("attention", {}).get("low_risk_scale", 0.5)),
        high_risk_scale=float(cfg.get("attention", {}).get("high_risk_scale", 1.0)),
    )
    logger = JsonlRunLogger(path=str(cfg.get("logging", {}).get("jsonl_path", "outputs/realtime_log.jsonl")))

    writer = None
    if bool(cfg.get("io", {}).get("save_video", False)):
        out_path = str(cfg.get("io", {}).get("output_video_path", "outputs/annotated.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    try:
        for captured in capture.stream():
            t0 = time.perf_counter()
            per_out = perception.infer(captured.bgr)
            t1 = time.perf_counter()

            # Use the latest available hazards to bias memory persistence. First pass uses empty hazards.
            mem = memory.update(
                timestamp=captured.timestamp,
                detections=per_out.detections,
                embeddings=per_out.embeddings,
                hazards=[],
            )
            t2 = time.perf_counter()

            hois = hoi.predict(
                timestamp=captured.timestamp,
                detections=per_out.detections,
                embeddings=per_out.embeddings,
                memory=mem,
            )
            t3 = time.perf_counter()

            hazard_out = reasoner.infer(hois)
            # Update once more with new hazard signals.
            mem = memory.update(
                timestamp=captured.timestamp,
                detections=per_out.detections,
                embeddings=per_out.embeddings,
                hazards=hazard_out.hazards,
            )
            t4 = time.perf_counter()

            frame_data = FrameData(
                timestamp=captured.timestamp,
                frame_index=captured.frame_index,
                frame_bgr=torch.from_numpy(captured.bgr.copy()),
                detections=per_out.detections,
                masks=per_out.masks,
                embeddings=per_out.embeddings,
                hois=hois,
                hazards=hazard_out.hazards,
                memory=mem,
                latency_ms={
                    "perception": (t1 - t0) * 1000.0,
                    "memory": (t2 - t1 + t4 - t3) * 1000.0,
                    "hoi": (t3 - t2) * 1000.0,
                    "hazard": (t4 - t3) * 1000.0,
                    "total": (t4 - t0) * 1000.0,
                },
            )

            allocation = attention.allocation(frame_data.detections, frame_data.hazards)
            logger.write(frame_data=frame_data, alerts=hazard_out.alerts, attention=allocation)
            annotated = render_frame(captured.bgr, frame_data, alerts=hazard_out.alerts)

            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("RiskSense-VLA Realtime", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and captured.frame_index + 1 >= args.max_frames:
                break
    finally:
        capture.close()
        logger.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
