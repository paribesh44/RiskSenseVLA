#!/usr/bin/env python3
"""Verify integrity and coverage of a generated synthetic hazard dataset."""

from __future__ import annotations

import argparse
import logging
import json
import sys
from pathlib import Path

import cv2

from risksense_vla.synthetic.scene_config import HAZARD_TEMPLATES

_LOG = logging.getLogger(__name__)

REQUIRED_FRAME_KEYS = {"frame_id", "frame_idx", "objects", "hoi", "hazard_score"}
REQUIRED_HOI_KEYS = {"subject", "action", "object", "confidence"}
REQUIRED_HAZARD_KEYS = {"subject", "action", "object", "score", "severity", "explanation"}
VALID_SEVERITIES = {"low", "medium", "high"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify synthetic hazard dataset")
    p.add_argument("--dataset-dir", default="data/synthetic")
    return p.parse_args()


class Verifier:
    def __init__(self, dataset_dir: str) -> None:
        self.root = Path(dataset_dir)
        self.annotations_path = self.root / "annotations.jsonl"
        self.frames_root = self.root / "frames"
        self.videos_root = self.root / "videos"
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.records: list[dict] = []

    def run_all(self) -> bool:
        self._load_annotations()
        if not self.records:
            self.errors.append("No annotation records found")
            return False
        self._check_schema()
        self._check_frame_files()
        self._check_video_files()
        self._check_hazard_coverage()
        self._check_hoi_consistency()
        self._check_score_ranges()
        return len(self.errors) == 0

    def _load_annotations(self) -> None:
        if not self.annotations_path.exists():
            self.errors.append(f"Annotations file not found: {self.annotations_path}")
            return
        for line_no, line in enumerate(
            self.annotations_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                self.records.append(json.loads(line))
            except json.JSONDecodeError:
                self.errors.append(f"Invalid JSON at line {line_no}")

    def _check_schema(self) -> None:
        for rec in self.records:
            sid = rec.get("scene_id", "<unknown>")
            if "scene_id" not in rec:
                self.errors.append("Record missing 'scene_id'")
            frames = rec.get("frames", [])
            if not isinstance(frames, list) or not frames:
                self.errors.append(f"{sid}: empty or missing 'frames' list")
                continue
            for frame in frames:
                self._validate_frame_schema(sid, frame)

    def _validate_frame_schema(self, sid: str, frame: dict) -> None:
        fid = frame.get("frame_idx", "?")
        missing = REQUIRED_FRAME_KEYS - set(frame.keys())
        if missing:
            self.errors.append(f"{sid} frame {fid}: missing {missing}")
        self._check_sub_keys(sid, fid, frame.get("hoi", {}), REQUIRED_HOI_KEYS, "hoi")
        self._check_sub_keys(sid, fid, frame.get("hazard_score", {}), REQUIRED_HAZARD_KEYS, "hazard_score")

    def _check_sub_keys(self, sid: str, fid: object, data: object, required: set[str], label: str) -> None:
        if not isinstance(data, dict):
            return
        missing = required - set(data.keys())
        if missing:
            self.errors.append(f"{sid} frame {fid}: {label} missing {missing}")

    def _check_frame_files(self) -> None:
        if not self.frames_root.exists():
            self.errors.append(f"Frames directory not found: {self.frames_root}")
            return
        for rec in self.records:
            sid = rec.get("scene_id", "")
            frames = rec.get("frames", [])
            angles = rec.get("camera_angles", ["front"])
            for angle in angles:
                dir_name = f"{sid}_{angle}" if angle != "front" else sid
                frame_dir = self.frames_root / dir_name
                if not frame_dir.exists():
                    self.errors.append(f"Frame dir missing: {frame_dir}")
                    continue
                pngs = sorted(frame_dir.glob("frame_*.png"))
                if len(pngs) != len(frames):
                    self.errors.append(
                        f"{dir_name}: expected {len(frames)} PNGs, found {len(pngs)}"
                    )

    def _check_video_files(self) -> None:
        if not self.videos_root.exists():
            self.errors.append(f"Videos directory not found: {self.videos_root}")
            return
        for rec in self.records:
            sid = rec.get("scene_id", "")
            frames = rec.get("frames", [])
            angles = rec.get("camera_angles", ["front"])
            for angle in angles:
                dir_name = f"{sid}_{angle}" if angle != "front" else sid
                mp4 = self.videos_root / f"{dir_name}.mp4"
                if not mp4.exists():
                    self.errors.append(f"Video missing: {mp4}")
                    continue
                cap = cv2.VideoCapture(str(mp4))
                vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if vid_frames != len(frames):
                    self.warnings.append(
                        f"{dir_name}: MP4 has {vid_frames} frames, annotation has {len(frames)}"
                    )

    def _check_hazard_coverage(self) -> None:
        seen: set[str] = set()
        for rec in self.records:
            event = rec.get("hazard_event", "")
            if event:
                seen.add(event)
        all_types = set(HAZARD_TEMPLATES.keys())
        missing = all_types - seen
        if missing:
            self.warnings.append(f"Hazard types not represented: {missing}")

    def _check_hoi_consistency(self) -> None:
        for rec in self.records:
            sid = rec.get("scene_id", "")
            for frame in rec.get("frames", []):
                hoi = frame.get("hoi", {})
                if not isinstance(hoi, dict):
                    continue
                obj_labels = {o.get("label") for o in frame.get("objects", [])}
                subj = hoi.get("subject", "")
                obj = hoi.get("object", "")
                if subj and subj not in obj_labels:
                    self.warnings.append(
                        f"{sid} f{frame.get('frame_idx', '?')}: HOI subject '{subj}' not in objects"
                    )
                if obj and obj not in obj_labels:
                    self.warnings.append(
                        f"{sid} f{frame.get('frame_idx', '?')}: HOI object '{obj}' not in objects"
                    )

    def _check_score_ranges(self) -> None:
        for rec in self.records:
            sid = rec.get("scene_id", "")
            for frame in rec.get("frames", []):
                hs = frame.get("hazard_score", {})
                if not isinstance(hs, dict):
                    continue
                score = hs.get("score", 0)
                if not (0.0 <= score <= 1.0):
                    self.errors.append(
                        f"{sid} f{frame.get('frame_idx', '?')}: score {score} out of [0,1]"
                    )
                sev = hs.get("severity", "")
                if sev not in VALID_SEVERITIES:
                    self.errors.append(
                        f"{sid} f{frame.get('frame_idx', '?')}: invalid severity '{sev}'"
                    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    v = Verifier(args.dataset_dir)
    v.run_all()

    if v.warnings:
        _LOG.warning("\nWARNINGS (%s):", len(v.warnings))
        for w in v.warnings:
            _LOG.warning("  [WARN] %s", w)

    if v.errors:
        _LOG.error("\nERRORS (%s):", len(v.errors))
        for e in v.errors:
            _LOG.error("  [FAIL] %s", e)
        _LOG.error("\nVerification FAILED (%s errors)", len(v.errors))
        sys.exit(1)
    else:
        _LOG.info("\nVerification PASSED (%s records checked, %s warnings)", len(v.records), len(v.warnings))


if __name__ == "__main__":
    main()
