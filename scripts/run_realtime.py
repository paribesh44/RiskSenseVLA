#!/usr/bin/env python3
"""End-to-end real-time inference loop for RiskSense-VLA."""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import torch

from risksense_vla.attention import SemanticAttentionScheduler
from risksense_vla.config import load_config
from risksense_vla.hazard import DistilledHazardReasoner
from risksense_vla.hoi import PredictiveHOIModule
from risksense_vla.io import VideoInput, resolve_source
from risksense_vla.memory import HazardAwareMemory
from risksense_vla.perception import OpenVocabPerception
from risksense_vla.runtime import pick_backend
from risksense_vla.types import FrameData
from risksense_vla.viz import JsonlRunLogger, render_frame

_LOG = logging.getLogger(__name__)


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
    _LOG.info("[RiskSense-VLA] backend=%s device=%s", backend.name, backend.device)

    src = args.source if args.source is not None else str(cfg.get("io", {}).get("source", 0))
    width = int(cfg.get("io", {}).get("width", 1280))
    height = int(cfg.get("io", {}).get("height", 720))
    fps = int(cfg.get("runtime", {}).get("target_fps", 25))
    capture = VideoInput(resolve_source(src), width=width, height=height, target_fps=fps)

    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
    memory = HazardAwareMemory()
    hoi = PredictiveHOIModule(
        future_horizon_seconds=int(cfg.get("hazard", {}).get("future_horizon_seconds", 3)),
        emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
    )
    reasoner = DistilledHazardReasoner(
        alert_threshold=float(cfg.get("hazard", {}).get("alert_threshold", 0.65)),
        checkpoint_path=str(cfg.get("hazard", {}).get("reasoner_checkpoint", "artifacts/hazard_reasoner.pt")),
        fallback_mode=str(cfg.get("hazard", {}).get("reasoner_fallback_mode", "blend")),
        emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
        backend_type=str(cfg.get("hazard", {}).get("backend_type", "tiny")),
        max_tokens=int(cfg.get("hazard", {}).get("max_tokens", 64)),
        temperature=float(cfg.get("hazard", {}).get("temperature", 0.2)),
        quantized=bool(cfg.get("hazard", {}).get("quantized", True)),
        lightweight_mode=bool(cfg.get("hazard", {}).get("lightweight_mode", False)),
        phi4_model_id=str(cfg.get("hazard", {}).get("phi4_model_id", "microsoft/Phi-4-multimodal-instruct")),
        phi4_precision=str(cfg.get("hazard", {}).get("phi4_precision", "int8")),
        phi4_estimated_vram_gb=float(cfg.get("hazard", {}).get("phi4_estimated_vram_gb", 10.0)),
        vlm_model_id=str(
            cfg.get("hazard", {}).get("vlm_model_id", "HuggingFaceTB/SmolVLM-500M-Instruct")
        ),
        explain=bool(cfg.get("hazard", {}).get("explain", True)),
        debug_prompt=bool(cfg.get("hazard", {}).get("debug_prompt", False)),
    )
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
            detections = perception.infer(captured.bgr)
            t1 = time.perf_counter()

            # Use the latest available hazards to bias memory persistence. First pass uses empty hazards.
            mem = memory.update(
                timestamp=captured.timestamp,
                detections=detections,
                hazards=None,
            )
            t2 = time.perf_counter()

            hoi_out = hoi.infer(
                memory_state=mem,
                object_detections=detections,
                timestamp=captured.timestamp,
            )
            hoi_triplets = hoi_out.as_triplets()
            t3 = time.perf_counter()

            hazard_out = reasoner.predict_hazard(
                hoi_current=hoi_out.hoi_current,
                hoi_future_embeddings=hoi_out.hoi_future_embeddings,
                memory_state=mem,
                frame_bgr=captured.bgr,
            )
            t_h = time.perf_counter()
            # Update once more with new hazard signals.
            mem = memory.update(
                timestamp=captured.timestamp,
                detections=detections,
                hazards=None,
                hazard_events=hazard_out.hazards,
            )
            t4 = time.perf_counter()

            frame_data = FrameData(
                timestamp=captured.timestamp,
                frame_index=captured.frame_index,
                frame_bgr=torch.from_numpy(captured.bgr.copy()),
                detections=detections,
                hois=hoi_triplets,
                hazards=hazard_out.hazards,
                memory=mem,
                latency_ms={
                    "perception": (t1 - t0) * 1000.0,
                    "memory": (t2 - t1 + t4 - t_h) * 1000.0,
                    "hoi": (t3 - t2) * 1000.0,
                    "hazard": (t_h - t3) * 1000.0,
                    "hazard_reasoner": hazard_out.inference_ms,
                    "total": (t4 - t0) * 1000.0,
                },
            )

            allocation = attention.allocation(frame_data.detections, frame_data.hazards)
            logger.write(
                frame_data=frame_data,
                alerts=hazard_out.alerts,
                attention=allocation,
                hazard_map=hazard_out.hazard_map,
                hazard_map_legacy=hazard_out.hazard_map_legacy,
                hazard_explanations=hazard_out.explanations,
                hazard_prompt_debug=hazard_out.prompt_debug,
                hazard_inference_ms=hazard_out.inference_ms,
                hazard_backend=hazard_out.backend,
                hazard_backend_metadata=hazard_out.backend_metadata,
            )
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
