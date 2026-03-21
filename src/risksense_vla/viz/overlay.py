"""Real-time visualization and JSONL logging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TextIO

import cv2
import numpy as np

from risksense_vla.types import FrameData, HazardScore, dataclass_to_json_ready


def _hazard_color(severity: str) -> tuple[int, int, int]:
    if severity == "high":
        return (0, 0, 255)
    if severity == "medium":
        return (0, 165, 255)
    return (0, 255, 0)


def _label_hazard(label: str, hazards: list[HazardScore]) -> HazardScore | None:
    rel = [h for h in hazards if h.object == label]
    if not rel:
        return None
    return max(rel, key=lambda x: x.score)


def render_frame(frame_bgr: np.ndarray, frame_data: FrameData, alerts: list[str]) -> np.ndarray:
    out = frame_bgr.copy()
    for det in frame_data.detections:
        hz = _label_hazard(det.label, frame_data.hazards)
        sev = hz.severity if hz else "low"
        score = hz.score if hz else 0.0
        color = _hazard_color(sev)
        x1, y1, x2, y2 = det.bbox_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{det.label} {det.confidence:.2f} hz={score:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

    y = 20
    for hoi in frame_data.hois[:8]:
        prefix = "PRED" if hoi.predicted else "HOI"
        text = f"{prefix}: {hoi.subject}-{hoi.action}-{hoi.object} ({hoi.confidence:.2f})"
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    for alert in alerts[:4]:
        cv2.putText(out, alert, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
        y += 20

    # Compact top-right panel for real-time hazard situational awareness.
    panel_w, panel_h = 340, 120
    px1 = max(0, out.shape[1] - panel_w - 10)
    py1 = 10
    px2 = min(out.shape[1] - 1, px1 + panel_w)
    py2 = min(out.shape[0] - 1, py1 + panel_h)
    panel = out[py1:py2, px1:px2]
    if panel.size:
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (32, 32, 32), -1)
        cv2.addWeighted(overlay, 0.55, panel, 0.45, 0.0, panel)

    top_hazard = max(frame_data.hazards, key=lambda h: h.score, default=None)
    status_text = "ALERT" if alerts else "MONITOR"
    status_color = (0, 0, 255) if alerts else (0, 255, 255)
    lines = [
        f"status: {status_text}",
        f"hazards: {len(frame_data.hazards)}",
        f"alerts: {len(alerts)}",
        (
            f"top: {top_hazard.object} {top_hazard.score:.2f} ({top_hazard.severity})"
            if top_hazard
            else "top: none"
        ),
    ]
    ty = py1 + 24
    cv2.putText(out, lines[0], (px1 + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
    for line in lines[1:]:
        ty += 22
        cv2.putText(out, line, (px1 + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    if alerts:
        banner_text = f"ACTIVE ALERT: {alerts[0]}"
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 180), -1)
        cv2.putText(out, banner_text[:120], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if frame_data.latency_ms:
        perf = " | ".join(f"{k}:{v:.1f}ms" for k, v in frame_data.latency_ms.items())
        cv2.putText(out, perf, (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return out


@dataclass(slots=True)
class JsonlRunLogger:
    """Structured JSONL logger with fixed per-frame schema."""

    path: str
    _fh: Optional[TextIO] = field(init=False, default=None)

    def __post_init__(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fh = p.open("a", encoding="utf-8")

    def write(
        self,
        frame_data: FrameData,
        alerts: list[str],
        attention: dict[str, float],
        hazard_map: dict[str, float] | None = None,
        hazard_map_legacy: dict[str, float] | None = None,
        hazard_explanations: dict[str, str] | None = None,
        hazard_prompt_debug: dict[str, str] | None = None,
        hazard_inference_ms: float | None = None,
        hazard_backend: str | None = None,
        hazard_backend_metadata: dict[str, object] | None = None,
    ) -> None:
        record = {
            "frame_id": frame_data.frame_index,
            "timestamp": frame_data.timestamp,
            "detections": [
                {
                    "track_id": det.track_id,
                    "label": det.label,
                    "confidence": float(det.confidence),
                    "bbox_xyxy": list(det.bbox_xyxy),
                    "mask_shape": list(det.mask.shape),
                    "clip_embedding_dim": int(det.clip_embedding.shape[0]),
                }
                for det in frame_data.detections
            ],
            "hois": dataclass_to_json_ready(frame_data.hois),
            "hazards": dataclass_to_json_ready(frame_data.hazards),
            "hazard_map": hazard_map or {},
            "hazard_map_legacy": hazard_map_legacy or {},
            "hazard_explanations": hazard_explanations or {},
            "hazard_prompt_debug": hazard_prompt_debug or {},
            "hazard_inference_ms": float(hazard_inference_ms) if hazard_inference_ms is not None else 0.0,
            "hazard_backend": hazard_backend or "unknown",
            "hazard_backend_metadata": hazard_backend_metadata or {},
            "memory_stats": (
                {
                    "num_objects": len(frame_data.memory.objects) if frame_data.memory else 0,
                    "state_norm": float(frame_data.memory.state_vector.norm().item())
                    if frame_data.memory
                    else 0.0,
                }
            ),
            "latency_ms": frame_data.latency_ms,
            "attention_allocation": attention,
            "alerts": alerts,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
