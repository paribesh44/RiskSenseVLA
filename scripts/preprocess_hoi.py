#!/usr/bin/env python3
"""Preprocess HOIGen/HICO-style annotations into temporal HOI sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--window", type=int, default=4, help="Temporal window in frames.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for sample in data.get("samples", []):
            frames = sample.get("frames", [])
            if not frames:
                continue
            for i in range(0, len(frames), max(1, args.window)):
                chunk = frames[i : i + args.window]
                rec = {
                    "video_id": sample.get("video_id", "unknown"),
                    "start_frame": i,
                    "end_frame": i + len(chunk) - 1,
                    "hois": [],
                }
                for frame in chunk:
                    for hoi in frame.get("hois", []):
                        rec["hois"].append(
                            {
                                "subject": hoi.get("subject", "human"),
                                "action": hoi.get("action", "interact"),
                                "object": hoi.get("object", "object"),
                                "frame_idx": frame.get("frame_idx", -1),
                            }
                        )
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
