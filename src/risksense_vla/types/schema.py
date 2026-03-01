"""Typed contracts for all cross-module payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class Detection:
    track_id: str
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    embedding_idx: int = -1


@dataclass(slots=True)
class HOITriplet:
    subject: str
    action: str
    object: str
    confidence: float
    t_start: float
    t_end: float
    predicted: bool = False


@dataclass(slots=True)
class HazardScore:
    subject: str
    action: str
    object: str
    score: float
    severity: str
    explanation: str


@dataclass(slots=True)
class MemoryObjectState:
    track_id: str
    label: str
    last_bbox_xyxy: tuple[int, int, int, int]
    persistence: float
    hazard_weight: float
    age_frames: int


@dataclass(slots=True)
class MemoryState:
    timestamp: float
    objects: list[MemoryObjectState] = field(default_factory=list)
    hoi_embedding: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 256))
    state_vector: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 512))


@dataclass(slots=True)
class FrameData:
    timestamp: float
    frame_index: int
    frame_bgr: torch.Tensor | None
    detections: list[Detection]
    masks: torch.Tensor
    embeddings: torch.Tensor
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
