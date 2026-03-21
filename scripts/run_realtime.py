#!/usr/bin/env python3
"""End-to-end real-time inference loop for RiskSense-VLA."""

from __future__ import annotations

import argparse
import logging
import random
import time
from collections import defaultdict

import cv2
import torch

from risksense_vla.attention import SemanticAttentionScheduler
from risksense_vla.config import load_config
from risksense_vla.eval.ablation import NaiveMemory
from risksense_vla.experimental import (
    apply_occlusion,
    get_bool,
    method_display_name,
    resolve_mode,
    seed_everything,
)
from risksense_vla.hazard import DistilledHazardReasoner
from risksense_vla.hoi import PredictiveHOIModule, ProtoHOIPredictor
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
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    seed_everything(seed)
    rng = random.Random(seed)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    _LOG.info("[RiskSense-VLA] backend=%s device=%s", backend.name, backend.device)

    src = args.source if args.source is not None else str(cfg.get("io", {}).get("source", 0))
    width = int(cfg.get("io", {}).get("width", 1280))
    height = int(cfg.get("io", {}).get("height", 720))
    fps = int(cfg.get("runtime", {}).get("target_fps", 25))
    capture = VideoInput(resolve_source(src), width=width, height=height, target_fps=fps)

    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
    memory_mode = resolve_mode(cfg, "memory_mode", "hazard_aware")
    hoi_mode = resolve_mode(cfg, "hoi_mode", "predictive")
    use_hazard_weighting = get_bool(cfg, "memory", "use_hazard_weighting", True)
    use_prediction = get_bool(cfg, "hoi", "use_prediction", True)
    use_vlm = get_bool(cfg, "hazard", "use_vlm", True)
    occlusion_prob = float(cfg.get("evaluation", {}).get("occlusion_prob", 0.0))

    if memory_mode == "naive":
        memory = NaiveMemory(emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)))
    else:
        memory = HazardAwareMemory(
            emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
            alpha=0.0 if not use_hazard_weighting else 0.14,
            beta=0.0 if not use_hazard_weighting else 1.4,
            use_hazard_weighting=use_hazard_weighting,
        )
    if hoi_mode == "frame_only" or not use_prediction:
        hoi: PredictiveHOIModule | ProtoHOIPredictor = ProtoHOIPredictor(
            future_horizon_seconds=int(cfg.get("hazard", {}).get("future_horizon_seconds", 3)),
            emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
        )
    else:
        hoi = PredictiveHOIModule(
        future_horizon_seconds=int(cfg.get("hazard", {}).get("future_horizon_seconds", 3)),
        emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
    )
    hazard_backend = str(cfg.get("hazard", {}).get("backend_type", "smolvlm"))
    hazard_lightweight = bool(cfg.get("hazard", {}).get("lightweight_mode", False))
    if not use_vlm:
        hazard_backend = "stub"
        hazard_lightweight = True
    reasoner = DistilledHazardReasoner(
        alert_threshold=float(cfg.get("hazard", {}).get("alert_threshold", 0.65)),
        checkpoint_path=str(cfg.get("hazard", {}).get("reasoner_checkpoint", "artifacts/hazard_reasoner.pt")),
        fallback_mode=str(cfg.get("hazard", {}).get("reasoner_fallback_mode", "blend")),
        emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
        backend_type=hazard_backend,
        max_tokens=int(cfg.get("hazard", {}).get("max_tokens", 64)),
        temperature=float(cfg.get("hazard", {}).get("temperature", 0.2)),
        quantized=bool(cfg.get("hazard", {}).get("quantized", True)),
        lightweight_mode=hazard_lightweight,
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
    pending_horizon_predictions: list[dict[str, object]] = []

    writer = None
    if bool(cfg.get("io", {}).get("save_video", False)):
        out_path = str(cfg.get("io", {}).get("output_video_path", "outputs/annotated.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    try:
        hazard_history: dict[str, list[float]] = defaultdict(list)
        lifetime_by_track: dict[str, int] = defaultdict(int)
        for captured in capture.stream():
            t0 = time.perf_counter()
            detections = perception.infer(captured.bgr)
            t1 = time.perf_counter()
            detections, occlusion_events = apply_occlusion(detections, occlusion_prob=occlusion_prob, rng=rng)

            # Use the latest available hazards to bias memory persistence. First pass uses empty hazards.
            mem = memory.update(
                timestamp=captured.timestamp,
                detections=detections,
                hazards=None,
            )
            t2 = time.perf_counter()

            if isinstance(hoi, PredictiveHOIModule):
                hoi_out = hoi.infer(
                    memory_state=mem,
                    object_detections=detections,
                    timestamp=captured.timestamp,
                )
                hoi_triplets = hoi_out.as_triplets()
                hoi_current = hoi_out.hoi_current
                hoi_future_embeddings = hoi_out.hoi_future_embeddings
            else:
                hoi_triplets = hoi.predict(
                    timestamp=captured.timestamp,
                    detections=detections,
                    memory=mem,
                )
                hoi_current = [h for h in hoi_triplets if not h.predicted]
                hoi_future_embeddings = torch.zeros((0, 0, 0), dtype=torch.float32)
            t3 = time.perf_counter()

            hazard_out = reasoner.predict_hazard(
                hoi_current=hoi_current,
                hoi_future_embeddings=hoi_future_embeddings,
                memory_state=mem,
                frame_bgr=captured.bgr,
            )
            t_h = time.perf_counter()
            # Update once more with new hazard signals.
            mem = memory.update(
                timestamp=captured.timestamp,
                detections=detections,
                hazards=None,
                hazard_events=hazard_out.hazards if use_hazard_weighting else None,
            )
            t4 = time.perf_counter()
            for track_id, score in hazard_out.hazard_map.items():
                hazard_history[track_id].append(float(score))
            for obj in mem.objects:
                lifetime_by_track[obj.track_id] = max(lifetime_by_track[obj.track_id], int(obj.age_frames))

            frame_data = FrameData(
                timestamp=captured.timestamp,
                frame_index=captured.frame_index,
                frame_bgr=torch.from_numpy(captured.bgr.copy()),
                detections=detections,
                hois=hoi_triplets,
                hazards=hazard_out.hazards,
                memory=mem,
                latency_ms={
                    "perception_ms": (t1 - t0) * 1000.0,
                    "memory_ms": (t2 - t1 + t4 - t_h) * 1000.0,
                    "hoi_ms": (t3 - t2) * 1000.0,
                    "hazard_ms": (t_h - t3) * 1000.0,
                    "hazard_reasoner": hazard_out.inference_ms,
                    "total_ms": (t4 - t0) * 1000.0,
                },
            )

            allocation = attention.allocation(frame_data.detections, frame_data.hazards)
            horizon_predictions: list[dict[str, object]] = []
            horizon_actuals: list[dict[str, object]] = []
            object_label_by_track = {obj.track_id: obj.label for obj in mem.objects}
            predicted_by_track_horizon: dict[str, dict[int, tuple[str, float]]] = defaultdict(dict)
            for hoi_triplet in hoi_triplets:
                if not hoi_triplet.predicted:
                    continue
                track_id = str(hoi_triplet.object_track_id).strip()
                if not track_id:
                    continue
                horizon = int(round(float(hoi_triplet.t_end - hoi_triplet.t_start)))
                if horizon < 1:
                    continue
                existing = predicted_by_track_horizon[track_id].get(horizon)
                if existing is None or float(hoi_triplet.confidence) > float(existing[1]):
                    predicted_by_track_horizon[track_id][horizon] = (hoi_triplet.action, float(hoi_triplet.confidence))
            for track_id, by_horizon in predicted_by_track_horizon.items():
                for horizon, (pred_action, _) in by_horizon.items():
                    target_timestamp = float(captured.timestamp + float(horizon))
                    pred_entry = {
                        "track_id": track_id,
                        "horizon": int(horizon),
                        "horizon_seconds": int(horizon),
                        "predicted_action": pred_action,
                        "source_frame_id": captured.frame_index,
                        "target_frame_id": captured.frame_index + int(round(float(fps) * horizon)),
                        "source_timestamp": float(captured.timestamp),
                        "target_timestamp": target_timestamp,
                        "object_label": object_label_by_track.get(track_id, ""),
                    }
                    horizon_predictions.append(pred_entry)
                    pending_horizon_predictions.append(pred_entry)
            observed_by_track: dict[str, tuple[str, float]] = {}
            for hoi_triplet in hoi_triplets:
                if hoi_triplet.predicted:
                    continue
                track_id = str(hoi_triplet.object_track_id).strip()
                if not track_id:
                    continue
                current_best = observed_by_track.get(track_id)
                if current_best is None or float(hoi_triplet.confidence) > float(current_best[1]):
                    observed_by_track[track_id] = (hoi_triplet.action, float(hoi_triplet.confidence))
            due, remaining = [], []
            for pending in pending_horizon_predictions:
                if float(captured.timestamp) >= float(pending.get("target_timestamp", float("inf"))):
                    due.append(pending)
                else:
                    remaining.append(pending)
            pending_horizon_predictions = remaining
            for pending in due:
                track_id = str(pending.get("track_id", "")).strip()
                horizon_actuals.append(
                    {
                        "track_id": track_id,
                        "object_label": pending.get("object_label", ""),
                        "source_frame_id": pending.get("source_frame_id", captured.frame_index),
                        "target_frame_id": captured.frame_index,
                        "horizon_seconds": pending.get("horizon_seconds", 0),
                        "horizon": pending.get("horizon", pending.get("horizon_seconds", 0)),
                        "predicted_action": pending.get("predicted_action", ""),
                        "actual_action": observed_by_track.get(track_id, ("", 0.0))[0],
                        "source_timestamp": pending.get("source_timestamp", captured.timestamp),
                        "target_timestamp": captured.timestamp,
                    }
                )
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
                occlusion_events=occlusion_events,
                horizon_predictions=horizon_predictions,
                horizon_actuals=horizon_actuals,
                metadata={
                    "memory_mode": memory_mode,
                    "hoi_mode": hoi_mode,
                    "method_name": method_display_name(
                        memory_mode if memory_mode == "naive" else hoi_mode if hoi_mode == "frame_only" else "hazard_aware"
                    ),
                },
                track_metrics=[
                    {
                        "track_id": obj.track_id,
                        "frame_id": captured.frame_index,
                        "hazard_score": float(hazard_out.hazard_map.get(obj.track_id, 0.0)),
                        "hazard_history": hazard_history.get(obj.track_id, []),
                        "predicted_action": predicted_by_track_horizon.get(obj.track_id, {}).get(1, ("", 0.0))[0],
                        "actual_action": observed_by_track.get(obj.track_id, ("", 0.0))[0],
                        "object_lifetime": lifetime_by_track.get(obj.track_id, 0),
                        "memory_strength": float(obj.persistence),
                    }
                    for obj in mem.objects
                ],
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
