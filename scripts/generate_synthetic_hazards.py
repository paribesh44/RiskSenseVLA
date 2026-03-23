#!/usr/bin/env python3
"""Generate synthetic hazard dataset: frames, video clips, and annotations."""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

_LOG = logging.getLogger(__name__)

from risksense_vla.synthetic import (
    DatasetWriter,
    SequenceEngine,
    build_scene_configs,
    get_renderer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic hazard dataset generation pipeline",
    )
    p.add_argument("--num-scenes", type=int, default=100)
    p.add_argument("--output-dir", default="data/synthetic")
    p.add_argument(
        "--renderer",
        choices=["procedural", "sd"],
        default="procedural",
        help="Frame renderer backend (procedural or sd for Stable Diffusion)",
    )
    p.add_argument(
        "--resolution",
        default="640x480",
        help="WxH resolution for generated frames",
    )
    p.add_argument("--enable-multi-angle", action="store_true")
    p.add_argument(
        "--rare-event-ratio",
        type=float,
        default=0.3,
        help="Minimum fraction of scenes that contain rare hazard events",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--scene-config-yaml",
        default=None,
        help="Optional YAML file with explicit scene configurations",
    )
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-frames", type=int, default=24)
    p.add_argument(
        "--legacy-jsonl",
        default=None,
        help="Path for legacy hazards.jsonl (default: <output-dir>/hazards.jsonl)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    w, h = (int(x) for x in args.resolution.split("x"))
    resolution = (w, h)

    _LOG.info("Building %s scene configs ...", args.num_scenes)
    configs = build_scene_configs(
        num_scenes=args.num_scenes,
        rare_event_ratio=args.rare_event_ratio,
        resolution=resolution,
        enable_multi_angle=args.enable_multi_angle,
        num_frames=args.num_frames,
        fps=args.fps,
        config_yaml=args.scene_config_yaml,
        seed=args.seed,
    )

    renderer = get_renderer(args.renderer)
    engine = SequenceEngine(seed=args.seed)
    writer = DatasetWriter(args.output_dir, fps=args.fps)

    all_sequences = []
    hazard_counter: Counter[str] = Counter()
    total_frames = 0
    t0 = time.perf_counter()

    for idx, cfg in enumerate(configs):
        scene_id = f"synth_{idx:05d}"
        sequences = engine.generate(cfg, scene_id)
        for seq in sequences:
            frames_bgr = [
                renderer.render_frame(cfg, frame) for frame in seq.frames
            ]
            writer.write_sequence(seq, frames_bgr)
            all_sequences.append(seq)
            total_frames += len(frames_bgr)
            if cfg.hazard_templates:
                hazard_counter[cfg.hazard_templates[0]] += 1

        if (idx + 1) % 10 == 0 or idx + 1 == len(configs):
            elapsed = time.perf_counter() - t0
            _LOG.info("  [%s/%s] scenes generated (%.1fs)", idx + 1, len(configs), elapsed)

    legacy_path = args.legacy_jsonl or str(Path(args.output_dir) / "hazards.jsonl")
    writer.write_legacy_jsonl(all_sequences, legacy_path)

    elapsed = time.perf_counter() - t0
    _LOG.info("\nDone in %.1fs", elapsed)
    _LOG.info("  Scenes:       %s", len(all_sequences))
    _LOG.info("  Total frames: %s", total_frames)
    _LOG.info("  Output dir:   %s", args.output_dir)
    _LOG.info("  Legacy JSONL: %s", legacy_path)
    _LOG.info("  Hazard distribution:")
    for hazard, count in sorted(hazard_counter.items()):
        _LOG.info("    %s: %s", hazard, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
