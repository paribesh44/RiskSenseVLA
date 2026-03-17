#!/usr/bin/env python3
"""Run prompt-driven hazard reasoner on synthetic Phase 5-style scenarios."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

_LOG = logging.getLogger(__name__)
import torch

from risksense_vla.hazard import DistilledHazardReasoner
from risksense_vla.types import HOITriplet, MemoryObjectState, MemoryState, dataclass_to_json_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--checkpoint", default="artifacts/hazard_reasoner.pt")
    parser.add_argument("--backend-type", default="tiny", choices=["tiny", "stub"])
    parser.add_argument("--fallback-mode", default="blend", choices=["blend", "vlm_only", "prior_only"])
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--debug-prompt", action="store_true")
    parser.add_argument("--output-jsonl", default="outputs/hazard_reasoner_example.jsonl")
    return parser.parse_args()


def _memory_for_frame(frame_idx: int) -> MemoryState:
    oscillation = 0.15 * np.sin(frame_idx / 3.0)
    objects = [
        MemoryObjectState(
            track_id="trk_knife_01",
            label="knife",
            last_bbox_xyxy=(120, 60, 180, 150),
            persistence=float(max(0.05, min(1.0, 0.75 + oscillation))),
            hazard_weight=0.85,
            age_frames=frame_idx + 1,
        ),
        MemoryObjectState(
            track_id="trk_bottle_01",
            label="bottle",
            last_bbox_xyxy=(220, 110, 280, 190),
            persistence=0.65,
            hazard_weight=0.15,
            age_frames=frame_idx + 1,
        ),
        MemoryObjectState(
            track_id="trk_stove_01",
            label="stove",
            last_bbox_xyxy=(40, 40, 130, 170),
            persistence=0.92,
            hazard_weight=0.90,
            age_frames=frame_idx + 1,
        ),
    ]
    return MemoryState(
        timestamp=float(frame_idx),
        objects=objects,
        hoi_embedding=torch.linspace(0.0, 1.0, steps=256, dtype=torch.float32).unsqueeze(0),
        state_vector=torch.zeros((1, 512), dtype=torch.float32),
    )


def _hois_for_frame(frame_idx: int) -> list[HOITriplet]:
    hot_surface_conf = 0.78 + 0.18 * np.sin(frame_idx / 4.0)
    cut_conf = 0.62 + 0.22 * np.cos(frame_idx / 5.0)
    return [
        HOITriplet(
            subject="human",
            action="touch_hot_surface",
            object="stove",
            confidence=float(max(0.0, min(1.0, hot_surface_conf))),
            t_start=float(frame_idx),
            t_end=float(frame_idx),
            predicted=False,
        ),
        HOITriplet(
            subject="human",
            action="cut",
            object="knife",
            confidence=float(max(0.0, min(1.0, cut_conf))),
            t_start=float(frame_idx),
            t_end=float(frame_idx + 1),
            predicted=True,
        ),
        HOITriplet(
            subject="human",
            action="carry",
            object="bottle",
            confidence=0.42,
            t_start=float(frame_idx),
            t_end=float(frame_idx),
            predicted=False,
        ),
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reasoner = DistilledHazardReasoner(
        checkpoint_path=args.checkpoint,
        fallback_mode=args.fallback_mode,
        alert_threshold=0.65,
        backend_type=args.backend_type,
        max_tokens=64,
        temperature=0.2,
        quantized=bool(args.quantized),
        lightweight_mode=True,
        explain=True,
        debug_prompt=bool(args.debug_prompt),
    )

    with out_path.open("w", encoding="utf-8") as fh:
        for frame_idx in range(max(0, args.max_frames)):
            memory_state = _memory_for_frame(frame_idx)
            hoi_current = _hois_for_frame(frame_idx)
            frame_bgr = np.zeros((240, 320, 3), dtype=np.uint8)

            out = reasoner.predict_hazard(
                hoi_current=hoi_current,
                hoi_future_embeddings=torch.zeros((len(hoi_current), 3, 256), dtype=torch.float32),
                memory_state=memory_state,
                frame_bgr=frame_bgr,
            )
            record = {
                "frame_id": frame_idx,
                "timestamp": float(frame_idx),
                "hoi_current": dataclass_to_json_ready(hoi_current),
                "hazards": dataclass_to_json_ready(out.hazards),
                "hazard_map": out.hazard_map,
                "hazard_map_legacy": out.hazard_map_legacy,
                "hazard_explanations": out.explanations,
                "hazard_prompt_debug": out.prompt_debug,
                "hazard_inference_ms": out.inference_ms,
                "hazard_backend": out.backend,
                "hazard_backend_metadata": out.backend_metadata,
                "global_risk_score": out.global_risk_score,
                "alerts": out.alerts,
            }
            fh.write(json.dumps(record) + "\n")
            _LOG.info(
                "[hazard-example] frame=%s hazard_map=%s global_risk=%.2f infer_ms=%.2f alerts=%s",
                frame_idx, out.hazard_map, out.global_risk_score, out.inference_ms, len(out.alerts),
            )
            for track_id, score in sorted(out.hazard_map.items()):
                expl = out.explanations.get(track_id, "")
                _LOG.info("  track=%s score=%.2f explanation=%s", track_id, score, expl)
            for alert in out.alerts:
                _LOG.info("  - %s", alert)
            if args.debug_prompt:
                for hoi_key, prompt in out.prompt_debug.items():
                    _LOG.info("  prompt[%s]:", hoi_key)
                    _LOG.info("%s", prompt)

    _LOG.info("[hazard-example] wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

