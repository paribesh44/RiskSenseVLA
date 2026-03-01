#!/usr/bin/env python3
"""Preprocess COCO/OpenImages annotations into unified detection format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--source-name", default="coco")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    with out_path.open("w", encoding="utf-8") as f:
        for img in data.get("images", []):
            anns = anns_by_image.get(img["id"], [])
            objects = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                objects.append(
                    {
                        "label": categories.get(ann["category_id"], "unknown"),
                        "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                        "area": ann.get("area", w * h),
                    }
                )
            rec = {
                "source": args.source_name,
                "image_id": img["id"],
                "file_name": img.get("file_name", ""),
                "width": img.get("width", 0),
                "height": img.get("height", 0),
                "objects": objects,
            }
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
