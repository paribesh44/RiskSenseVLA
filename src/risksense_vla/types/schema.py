"""Typed contracts for all cross-module payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class Detection:
    """Legacy detection contract retained for adapter internals."""

    track_id: str
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    embedding_idx: int = -1
    embedding: torch.Tensor | None = None


@dataclass(slots=True)
class PerceptionDetection:
    """Canonical Phase 1 detection contract used across Phases 1-4: track, label, bbox, mask, CLIP embedding."""

    track_id: str
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: torch.Tensor
    clip_embedding: torch.Tensor


@dataclass(slots=True)
class HOITriplet:
    """Human-object interaction triplet: subject, action, object with confidence and time span."""

    subject: str
    action: str
    object: str
    confidence: float
    t_start: float
    t_end: float
    predicted: bool = False


@dataclass(slots=True)
class HazardScore:
    """Scored hazard with subject/action/object, severity, and explanation."""

    subject: str
    action: str
    object: str
    score: float
    severity: str
    explanation: str


@dataclass(slots=True)
class MemoryObjectState:
    """Tracked object state in memory: bbox, persistence, hazard weight, and age."""

    track_id: str
    label: str
    last_bbox_xyxy: tuple[int, int, int, int]
    persistence: float
    hazard_weight: float
    age_frames: int


@dataclass(slots=True)
class MemoryState:
    """Aggregate memory state: timestamp, tracked objects, HOI embedding, and state vector."""

    timestamp: float
    objects: list[MemoryObjectState] = field(default_factory=list)
    hoi_embedding: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 256))
    state_vector: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 512))


@dataclass(slots=True)
class FrameData:
    """Single-frame payload: timestamp, frame index, image, detections, HOIs, hazards, memory, latency."""

    timestamp: float
    frame_index: int
    frame_bgr: torch.Tensor | None
    detections: list[PerceptionDetection]
    hois: list[HOITriplet]
    hazards: list[HazardScore]
    memory: MemoryState | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)


def dataclass_to_json_ready(item: Any) -> Any:
    """Convert nested dataclass/tensors into JSON-safe objects."""
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().tolist()
    if hasattr(item, "__dataclass_fields__"):
        data = asdict(item)
        return {k: dataclass_to_json_ready(v) for k, v in data.items()}
    if isinstance(item, dict):
        return {k: dataclass_to_json_ready(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return [dataclass_to_json_ready(v) for v in item]
    return item
