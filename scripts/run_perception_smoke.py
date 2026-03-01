#!/usr/bin/env python3
"""Perception-only smoke run for webcam/video inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from risksense_vla.config import load_config
from risksense_vla.io import VideoInput, resolve_source
from risksense_vla.perception import OpenVocabPerception
from risksense_vla.runtime import pick_backend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default=None)
    p.add_argument("--source", default=None, help="Camera index or video file path.")
    p.add_argument("--labels", default="", help="Comma-separated open-vocab labels.")
    p.add_argument("--max-frames", type=int, default=120, help="0 means stream until source ends.")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--log-jsonl", default="outputs/perception_smoke.jsonl")
    return p.parse_args()


def _render(frame_bgr, detections, latency_ms: float):
    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(
            out,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        f"perception={latency_ms:.1f}ms",
        (10, out.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    print(f"[perception-smoke] backend={backend.name} device={backend.device}")

    src = args.source if args.source is not None else str(cfg.get("io", {}).get("source", 0))
    width = int(cfg.get("io", {}).get("width", 1280))
    height = int(cfg.get("io", {}).get("height", 720))
    fps = int(cfg.get("runtime", {}).get("target_fps", 25))
    capture = VideoInput(resolve_source(src), width=width, height=height, target_fps=fps)
    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()] if args.labels else None

    out_path = Path(args.log_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latencies_ms: list[float] = []

    try:
        with out_path.open("w", encoding="utf-8") as fh:
            for captured in capture.stream():
                t0 = time.perf_counter()
                out = perception.infer(captured.bgr, labels=labels)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt_ms)
                fps_now = 1000.0 / dt_ms if dt_ms > 0 else 0.0
                record = {
                    "frame_id": captured.frame_index,
                    "timestamp": captured.timestamp,
                    "num_detections": len(out.detections),
                    "labels": [d.label for d in out.detections],
                    "latency_ms": dt_ms,
                    "fps": fps_now,
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()

                if not args.no_display:
                    vis = _render(captured.bgr, out.detections, dt_ms)
                    cv2.imshow("RiskSense-VLA Perception Smoke", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if captured.frame_index % 20 == 0:
                    print(
                        f"[perception-smoke] frame={captured.frame_index} "
                        f"dets={len(out.detections)} latency={dt_ms:.1f}ms fps={fps_now:.1f}"
                    )
                if args.max_frames and captured.frame_index + 1 >= args.max_frames:
                    break
    finally:
        capture.close()
        cv2.destroyAllWindows()

    if latencies_ms:
        avg_ms = sum(latencies_ms) / len(latencies_ms)
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        print(
            f"[perception-smoke] completed frames={len(latencies_ms)} "
            f"avg_latency={avg_ms:.2f}ms avg_fps={avg_fps:.2f}"
        )


if __name__ == "__main__":
    main()
