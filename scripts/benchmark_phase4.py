#!/usr/bin/env python3
"""Benchmark strict Phase-4 runtime latency and FPS."""

from __future__ import annotations

import argparse
import json
import logging
import random
import copy
from pathlib import Path
import time
from collections import defaultdict

from risksense_vla.config import load_config
from risksense_vla.eval.ablation import NaiveMemory
from risksense_vla.experimental import (
    get_bool,
    method_display_name,
    resolve_mode,
    seed_everything,
)
from risksense_vla.eval.metrics import hazard_lead_time
from risksense_vla.hazard import DistilledHazardReasoner
from risksense_vla.hoi import PredictiveHOIModule, ProtoHOIPredictor
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


def _apply_occlusion_with_base_mask(
    detections: list[object],
    *,
    occlusion_prob: float,
    frame_index: int,
    base_mask: dict[str, float],
    rng: random.Random,
) -> tuple[list[object], list[dict[str, object]]]:
    if occlusion_prob <= 0.0 or not detections:
        return detections, []
    kept = []
    events: list[dict[str, object]] = []
    for det in detections:
        key = f"{frame_index}:{det.track_id}"
        sample = base_mask.get(key)
        if sample is None:
            sample = float(rng.random())
            base_mask[key] = sample
        if sample < occlusion_prob:
            events.append(
                {
                    "track_id": det.track_id,
                    "label": det.label,
                    "event": "dropped",
                    "sample": sample,
                    "occlusion_prob": float(occlusion_prob),
                }
            )
            continue
        kept.append(det)
    return kept, events


def _run_single_benchmark(
    cfg: dict[str, object],
    *,
    args: argparse.Namespace,
    backend: object,
    occlusion_prob: float,
    base_occlusion_mask: dict[str, float],
) -> dict[str, object]:
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    rng = random.Random(seed)
    source = args.source if args.source is not None else str(cfg.get("io", {}).get("source", 0))
    width = int(cfg.get("io", {}).get("width", 1280))
    height = int(cfg.get("io", {}).get("height", 720))
    fps = int(cfg.get("runtime", {}).get("target_fps", 25))
    capture = VideoInput(resolve_source(source), width=width, height=height, target_fps=fps)

    memory_mode = resolve_mode(cfg, "memory_mode", "hazard_aware")
    hoi_mode = resolve_mode(cfg, "hoi_mode", "predictive")
    use_hazard_weighting = get_bool(cfg, "memory", "use_hazard_weighting", True)
    use_prediction = get_bool(cfg, "hoi", "use_prediction", True)
    use_vlm = get_bool(cfg, "hazard", "use_vlm", True)

    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
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
    hazard_backend = str(cfg.get("hazard", {}).get("backend_type", "phi4_mm"))
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
        quantized=bool(cfg.get("hazard", {}).get("quantized", False)),
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

    per_ms: list[float] = []
    mem_ms: list[float] = []
    hoi_ms: list[float] = []
    hazard_ms: list[float] = []
    e2e_ms: list[float] = []
    horizon_hits: dict[int, int] = defaultdict(int)
    horizon_totals: dict[int, int] = defaultdict(int)
    pending_predictions: list[dict[str, object]] = []
    hazard_events: list[dict[str, object]] = []
    hazard_predictions: list[dict[str, object]] = []
    frame_count = 0

    try:
        for captured in capture.stream():
            if frame_count >= max(1, args.frames):
                break
            t0 = time.perf_counter()
            detections = perception.infer(captured.bgr)
            t1 = time.perf_counter()
            detections, _ = _apply_occlusion_with_base_mask(
                detections,
                occlusion_prob=occlusion_prob,
                frame_index=frame_count,
                base_mask=base_occlusion_mask,
                rng=rng,
            )
            mem = memory.update(timestamp=captured.timestamp, detections=detections, hazards=None)
            t2 = time.perf_counter()
            if isinstance(hoi, PredictiveHOIModule):
                hoi_out = hoi.infer(memory_state=mem, object_detections=detections, timestamp=captured.timestamp)
                hoi_current = hoi_out.hoi_current
                hoi_future_embeddings = hoi_out.hoi_future_embeddings
                hoi_triplets = hoi_out.as_triplets()
            else:
                hoi_triplets = hoi.predict(timestamp=captured.timestamp, detections=detections, memory=mem)
                hoi_current = [h for h in hoi_triplets if not h.predicted]
                hoi_future_embeddings = None
            t3 = time.perf_counter()
            hazard_out = reasoner.predict_hazard(
                hoi_current=hoi_current,
                hoi_future_embeddings=hoi_future_embeddings,
                memory_state=mem,
                frame_bgr=captured.bgr,
            )
            t4 = time.perf_counter()
            memory.update(
                timestamp=captured.timestamp,
                detections=detections,
                hazards=None,
                hazard_events=hazard_out.hazards if use_hazard_weighting else None,
            )
            t5 = time.perf_counter()
            predicted_by_track_horizon: dict[str, dict[int, tuple[str, float]]] = defaultdict(dict)
            observed_by_track: dict[str, tuple[str, float]] = {}
            for hoi_triplet in hoi_triplets:
                track_id = str(hoi_triplet.object_track_id).strip()
                if not track_id:
                    continue
                if hoi_triplet.predicted:
                    horizon = int(round(float(hoi_triplet.t_end - hoi_triplet.t_start)))
                    if horizon < 1:
                        continue
                    existing = predicted_by_track_horizon[track_id].get(horizon)
                    if existing is None or float(hoi_triplet.confidence) > float(existing[1]):
                        predicted_by_track_horizon[track_id][horizon] = (hoi_triplet.action, float(hoi_triplet.confidence))
                else:
                    current_best = observed_by_track.get(track_id)
                    if current_best is None or float(hoi_triplet.confidence) > float(current_best[1]):
                        observed_by_track[track_id] = (hoi_triplet.action, float(hoi_triplet.confidence))
            due, remaining = [], []
            for pending in pending_predictions:
                if float(captured.timestamp) >= float(pending.get("target_timestamp", float("inf"))):
                    due.append(pending)
                else:
                    remaining.append(pending)
            pending_predictions = remaining
            for pending in due:
                horizon = int(pending.get("horizon_seconds", 0))
                track_id = str(pending.get("track_id", "")).strip()
                actual_action = observed_by_track.get(track_id, ("", 0.0))[0]
                if actual_action:
                    horizon_totals[horizon] += 1
                    if str(pending.get("predicted_action", "")) == actual_action:
                        horizon_hits[horizon] += 1
            for track_id, by_horizon in predicted_by_track_horizon.items():
                for horizon, (action, _) in by_horizon.items():
                    pred_entry = {
                        "track_id": track_id,
                        "source_frame_id": frame_count,
                        "horizon_seconds": horizon,
                        "predicted_action": action,
                        "source_timestamp": float(captured.timestamp),
                        "target_timestamp": float(captured.timestamp + float(horizon)),
                    }
                    pending_predictions.append(pred_entry)
                    hazard_predictions.append(pred_entry)
            for hz in hazard_out.hazards:
                if float(hz.score) >= float(cfg.get("hazard", {}).get("alert_threshold", 0.65)):
                    hazard_events.append(
                        {
                            "frame_id": frame_count,
                            "track_id": hz.track_id,
                            "action": hz.action,
                        }
                    )

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
    lead = hazard_lead_time(hazard_events, hazard_predictions)
    report: dict[str, object] = {
        "backend": backend.name,
        "device": backend.device,
        "frames": frame_count,
        "occlusion": float(occlusion_prob),
        "perception_ms": _mean(per_ms),
        "memory_ms": _mean(mem_ms),
        "hoi_ms": _mean(hoi_ms),
        "hazard_ms": _mean(hazard_ms),
        "end_to_end_fps": float(end_to_end_fps),
        "min_fps_threshold": float(args.min_fps),
        "THC": 0.0,
        "prediction_accuracy@1s": float(horizon_hits[1] / horizon_totals[1]) if horizon_totals[1] else 0.0,
        "prediction_accuracy@2s": float(horizon_hits[2] / horizon_totals[2]) if horizon_totals[2] else 0.0,
        "prediction_accuracy@3s": float(horizon_hits[3] / horizon_totals[3]) if horizon_totals[3] else 0.0,
        "hazard_lead_time_mean": float(lead["mean"]),
        "hazard_lead_time_median": float(lead["median"]),
        "hazard_lead_time_distribution": lead["distribution"],
        "metadata": {
            "memory_mode": memory_mode,
            "hoi_mode": hoi_mode,
            "method_name": method_display_name(memory_mode if memory_mode == "naive" else hoi_mode if hoi_mode == "frame_only" else "hazard_aware"),
            "seed": seed,
        },
    }
    return report


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    seed_everything(seed)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    if args.require_gpu and backend.name not in {"mps", "cuda"}:
        raise RuntimeError("Phase-4 benchmark requires GPU backend (mps/cuda).")
    eval_cfg = cfg.get("evaluation", {})
    occlusion_prob = float(eval_cfg.get("occlusion_prob", 0.0))
    occlusion_levels_raw = eval_cfg.get("occlusion_levels")
    occlusion_levels = (
        [float(v) for v in occlusion_levels_raw]
        if isinstance(occlusion_levels_raw, list) and occlusion_levels_raw
        else [occlusion_prob]
    )
    results: list[dict[str, object]] = []
    base_occlusion_mask: dict[str, float] = {}
    for level in occlusion_levels:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("evaluation", {})
        run_cfg["evaluation"]["occlusion_prob"] = float(level)
        results.append(
            _run_single_benchmark(
                run_cfg,
                args=args,
                backend=backend,
                occlusion_prob=float(level),
                base_occlusion_mask=base_occlusion_mask,
            )
        )
    single_mode = len(occlusion_levels) == 1 and not (
        isinstance(occlusion_levels_raw, list) and len(occlusion_levels_raw) > 1
    )
    report: dict[str, object]
    if single_mode:
        report = results[0]
    else:
        report = {
            "backend": backend.name,
            "device": backend.device,
            "occlusion_levels": occlusion_levels,
            "results": results,
        }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _LOG.info("%s", json.dumps(report, indent=2))
    min_fps = min(float(r.get("end_to_end_fps", 0.0)) for r in results) if results else 0.0
    if min_fps < float(args.min_fps):
        raise SystemExit(
            f"Phase-4 benchmark failed: end_to_end_fps={min_fps:.2f} < threshold={args.min_fps:.2f}"
        )


if __name__ == "__main__":
    main()
