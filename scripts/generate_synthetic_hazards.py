#!/usr/bin/env python3
"""Generate synthetic hazard scenarios with temporal augmentations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


HAZARD_SEVERITY = {
    "hot_surface_contact": "high",
    "sharp_tool_contact": "high",
    "spill_risk": "medium",
    "trip_obstacle": "medium",
    "clutter": "low",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-scenes", type=int, default=100)
    p.add_argument("--output-jsonl", default="data/synthetic/hazards.jsonl")
    p.add_argument("--enable-multi-angle", action="store_true")
    return p.parse_args()


def apply_temporal_augmentations(seq: list[dict]) -> list[dict]:
    out = []
    frame_skip = random.choice([1, 1, 2, 3])  # variable frame-rate style.
    occlusion_prob = random.uniform(0.05, 0.2)
    for i, frame in enumerate(seq):
        if i % frame_skip != 0:
            continue
        if random.random() < occlusion_prob:
            frame = {**frame, "occluded": True}
        out.append(frame)
    return out


def build_scene(scene_id: int, multi_angle: bool) -> dict:
    event = random.choice(list(HAZARD_SEVERITY.keys()))
    severity = HAZARD_SEVERITY[event]
    base_frames = []
    for t in range(24):
        base_frames.append(
            {
                "frame_idx": t,
                "event": event,
                "severity": severity,
                "objects": ["person", "kitchen_counter", "knife" if "sharp" in event else "stove"],
                "hoi": {"subject": "human", "action": "interact", "object": "hazard_object"},
            }
        )
    frames = apply_temporal_augmentations(base_frames)
    scene = {
        "scene_id": f"synth_{scene_id:05d}",
        "hazard_event": event,
        "hazard_severity": severity,
        "frames": frames,
        "camera_angles": ["front"],
    }
    if multi_angle:
        scene["camera_angles"] = ["front", "left", "right"]
    return scene


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_scenes):
            scene = build_scene(i, args.enable_multi_angle)
            f.write(json.dumps(scene) + "\n")
    print(f"Wrote {args.num_scenes} synthetic hazard scenes to {out_path}")


if __name__ == "__main__":
    main()
