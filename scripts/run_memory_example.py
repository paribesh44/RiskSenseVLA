#!/usr/bin/env python3
"""Example sequential hazard-memory updates with optional logging."""

from __future__ import annotations

import argparse
import json
import logging

import torch

from risksense_vla.memory import HazardAwareMemory
from risksense_vla.types import PerceptionDetection

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--log", action="store_true", help="Print per-frame memory summary.")
    return parser.parse_args()


def _make_detection(track_id: str, label: str, confidence: float, bbox_xyxy: tuple[int, int, int, int], emb_seed: int) -> PerceptionDetection:
    emb = torch.linspace(0.0, 1.0, steps=256, dtype=torch.float32)
    emb = torch.roll(emb, shifts=int(emb_seed % 256))
    x1, y1, x2, y2 = bbox_xyxy
    mask = torch.zeros((240, 320), dtype=torch.float32)
    mask[y1:y2, x1:x2] = 1.0
    return PerceptionDetection(
        track_id=track_id,
        label=label,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
        mask=mask,
        clip_embedding=emb,
    )


def _frame_detections(frame_idx: int) -> tuple[list[PerceptionDetection], list[float]]:
    detections: list[PerceptionDetection] = []
    hazard_scores: list[float] = []
    if frame_idx < 9:
        detections.append(
            _make_detection(
                track_id="person-1",
                label="person",
                confidence=0.95,
                bbox_xyxy=(40 + frame_idx, 60, 120 + frame_idx, 220),
                emb_seed=frame_idx,
            )
        )
        hazard_scores.append(0.35)
    if 2 <= frame_idx <= 10:
        detections.append(
            _make_detection(
                track_id="knife-2",
                label="knife",
                confidence=0.90,
                bbox_xyxy=(180, 100 + frame_idx, 230, 180 + frame_idx),
                emb_seed=frame_idx + 10,
            )
        )
        hazard_scores.append(0.92)
    if frame_idx in {0, 1, 5, 6, 7}:
        detections.append(
            _make_detection(
                track_id="bottle-3",
                label="bottle",
                confidence=0.82,
                bbox_xyxy=(260, 120, 300, 200),
                emb_seed=frame_idx + 20,
            )
        )
        hazard_scores.append(0.15)
    return detections, hazard_scores


def main() -> None:
    args = parse_args()
    memory = HazardAwareMemory(log_updates=False)
    timestamp = 0.0
    for frame_idx in range(max(1, args.frames)):
        detections, hazard_scores = _frame_detections(frame_idx)
        state = memory.update(
            timestamp=timestamp,
            detections=detections,
            hazards=hazard_scores if hazard_scores else None,
            log_callback=(
                (lambda payload: _LOG.info("%s", json.dumps(payload))) if args.log else None
            ),
        )
        _LOG.info(
            "frame=%02d objects=%02d avg_p=%.3f",
            frame_idx,
            len(state.objects),
            memory.summary()["avg_persistence"],
        )
        timestamp += 1.0 / 25.0


if __name__ == "__main__":
    main()
