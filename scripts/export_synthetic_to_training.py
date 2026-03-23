#!/usr/bin/env python3
"""Export synthetic hazard dataset to training-ready formats."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert synthetic annotations to training formats",
    )
    p.add_argument("--dataset-dir", default="data/synthetic")
    p.add_argument("--output-dir", default="data/training")
    p.add_argument(
        "--formats",
        default="hoigen,temporal,hazard_vlm",
        help="Comma-separated list: hoigen, temporal, hazard_vlm",
    )
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temporal-window", type=int, default=4)
    return p.parse_args()


def load_records(dataset_dir: str) -> list[dict]:
    path = Path(dataset_dir) / "annotations.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Annotations not found: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def stratified_split(
    records: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split records into train/val, stratified by hazard_event."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_type[rec.get("hazard_event", "unknown")].append(rec)

    train, val = [], []
    for hazard_type, recs in by_type.items():
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * val_ratio))
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])
    return train, val


def export_hoigen(records: list[dict], out_path: Path) -> None:
    """Export to HOIGen-style JSON (compatible with HOIGenRawDataset)."""
    samples = []
    for rec in records:
        frames_out = []
        for frame in rec.get("frames", []):
            hoi = frame.get("hoi", {})
            hois = [
                {
                    "subject": hoi.get("subject", "human"),
                    "action": hoi.get("action", "interact"),
                    "object": hoi.get("object", "object"),
                }
            ]
            frames_out.append({"frame_idx": frame.get("frame_idx", 0), "hois": hois})
        samples.append({"video_id": rec.get("scene_id", ""), "frames": frames_out})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"samples": samples}, indent=2), encoding="utf-8"
    )


def export_temporal_jsonl(
    records: list[dict], out_path: Path, window: int = 4
) -> None:
    """Export to temporal JSONL (compatible with TemporalHOIPreprocessedDataset)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            frames = rec.get("frames", [])
            video_id = rec.get("scene_id", "")
            for start in range(0, len(frames), window):
                chunk = frames[start : start + window]
                if not chunk:
                    continue
                hois = []
                for frame in chunk:
                    hoi = frame.get("hoi", {})
                    hois.append(
                        {
                            "subject": hoi.get("subject", "human"),
                            "action": hoi.get("action", "interact"),
                            "object": hoi.get("object", "object"),
                            "frame_idx": frame.get("frame_idx", 0),
                        }
                    )
                entry = {
                    "video_id": video_id,
                    "start_frame": chunk[0].get("frame_idx", 0),
                    "end_frame": chunk[-1].get("frame_idx", 0),
                    "hois": hois,
                }
                f.write(json.dumps(entry) + "\n")


def export_hazard_vlm_jsonl(records: list[dict], out_path: Path) -> None:
    """Export to hazard VLM JSONL (compatible with train_hazard_vlm.py)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            entry = {
                "scene_id": rec.get("scene_id", ""),
                "hazard_event": rec.get("hazard_event", ""),
                "hazard_severity": rec.get("hazard_severity", "low"),
                "frames": rec.get("frames", []),
                "camera_angles": rec.get("camera_angles", ["front"]),
            }
            f.write(json.dumps(entry) + "\n")


EXPORTERS = {
    "hoigen": lambda recs, out, **kw: export_hoigen(recs, out / "hoigen.json"),
    "temporal": lambda recs, out, **kw: export_temporal_jsonl(
        recs, out / "temporal.jsonl", window=kw.get("window", 4)
    ),
    "hazard_vlm": lambda recs, out, **kw: export_hazard_vlm_jsonl(
        recs, out / "hazard_vlm.jsonl"
    ),
}


def main() -> None:
    args = parse_args()
    records = load_records(args.dataset_dir)
    _LOG.info("Loaded %d annotation records from %s", len(records), args.dataset_dir)

    train_recs, val_recs = stratified_split(records, args.val_ratio, args.seed)
    _LOG.info("  Train: %d  Val: %d", len(train_recs), len(val_recs))

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_root = Path(args.output_dir)

    for fmt in formats:
        exporter = EXPORTERS.get(fmt)
        if exporter is None:
            _LOG.warning("  Unknown format '%s', skipping", fmt)
            continue
        for split_name, split_recs in [("train", train_recs), ("val", val_recs)]:
            split_dir = out_root / split_name
            exporter(split_recs, split_dir, window=args.temporal_window)
            _LOG.info("  Wrote %s -> %s/", fmt, split_dir)

    _LOG.info("Export complete.")


if __name__ == "__main__":
    main()
