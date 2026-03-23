"""Dataset writer: PNG frames, MP4 clips, and JSONL annotations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from risksense_vla.synthetic.sequence_engine import AnnotatedSequence

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Writes rendered frames, video clips, and structured annotations."""

    def __init__(self, output_root: str | Path, *, fps: int = 24) -> None:
        self.output_root = Path(output_root)
        self.fps = fps
        self._annotations_path = self.output_root / "annotations.jsonl"
        self._frames_root = self.output_root / "frames"
        self._videos_root = self.output_root / "videos"

    def write_sequence(
        self,
        sequence: AnnotatedSequence,
        frames_bgr: list[np.ndarray],
    ) -> None:
        """Persist a full sequence: PNG frames, MP4 clip, annotation record."""
        sid = sequence.scene_id
        angle = sequence.camera_angle
        dir_name = f"{sid}_{angle}" if angle != "front" else sid

        self._write_frames(dir_name, frames_bgr)
        self._write_video(dir_name, frames_bgr, sequence.scene_config.fps)
        self._write_annotation(sequence, dir_name)

    def _write_frames(self, dir_name: str, frames_bgr: list[np.ndarray]) -> None:
        frame_dir = self._frames_root / dir_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        for idx, bgr in enumerate(frames_bgr):
            path = frame_dir / f"frame_{idx:04d}.png"
            cv2.imwrite(str(path), bgr)

    def _write_video(
        self, dir_name: str, frames_bgr: list[np.ndarray], fps: int
    ) -> None:
        self._videos_root.mkdir(parents=True, exist_ok=True)
        if not frames_bgr:
            return
        h, w = frames_bgr[0].shape[:2]
        out_path = self._videos_root / f"{dir_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        try:
            for bgr in frames_bgr:
                writer.write(bgr)
        finally:
            writer.release()

    def _write_annotation(
        self, sequence: AnnotatedSequence, dir_name: str
    ) -> None:
        self._annotations_path.parent.mkdir(parents=True, exist_ok=True)
        hazard_type = ""
        hazard_severity = "low"
        if sequence.scene_config.hazard_templates:
            from risksense_vla.synthetic.scene_config import HAZARD_TEMPLATES

            hazard_type = sequence.scene_config.hazard_templates[0]
            hazard_severity = HAZARD_TEMPLATES.get(hazard_type, "low")

        frame_records: list[dict] = []
        for frame in sequence.frames:
            frame_records.append(
                {
                    "frame_id": f"{dir_name}_f{frame.frame_idx:04d}",
                    "frame_idx": frame.frame_idx,
                    "objects": frame.objects,
                    "hoi": frame.hoi,
                    "hazard_score": frame.hazard_score,
                    "occluded": frame.occluded,
                }
            )

        record = {
            "scene_id": sequence.scene_id,
            "hazard_event": hazard_type,
            "hazard_severity": hazard_severity,
            "room_type": sequence.scene_config.room_type,
            "camera_angles": sequence.scene_config.camera_angles,
            "frames": frame_records,
        }

        with self._annotations_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def write_legacy_jsonl(
        self, sequences: list[AnnotatedSequence], path: str | Path
    ) -> None:
        """Write hazards.jsonl in the legacy format consumed by train_hazard_vlm.py."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for seq in sequences:
                hazard_type = ""
                hazard_severity = "low"
                if seq.scene_config.hazard_templates:
                    from risksense_vla.synthetic.scene_config import HAZARD_TEMPLATES

                    hazard_type = seq.scene_config.hazard_templates[0]
                    hazard_severity = HAZARD_TEMPLATES.get(hazard_type, "low")

                legacy_frames = []
                for frame in seq.frames:
                    legacy_frames.append(
                        {
                            "frame_idx": frame.frame_idx,
                            "event": hazard_type,
                            "severity": hazard_severity,
                            "objects": [o["label"] for o in frame.objects],
                            "hoi": frame.hoi,
                            "occluded": frame.occluded,
                        }
                    )
                record = {
                    "scene_id": seq.scene_id,
                    "hazard_event": hazard_type,
                    "hazard_severity": hazard_severity,
                    "frames": legacy_frames,
                    "camera_angles": seq.scene_config.camera_angles,
                }
                f.write(json.dumps(record) + "\n")
