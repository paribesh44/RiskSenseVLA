"""Backward-compatible imports for the hazard reasoner module."""

from .backends import BaseVLMBackend, HazardConfig, Phi4MultimodalBackend, StubBackend, TinyLocalVLMBackend, VLMOutput
from .hazard_reasoner import (
    DistilledHazardReasoner,
    HOIKey,
    HazardOutput,
    HazardReasoner,
    HazardReasoningOutput,
    LaCHazardReasoner,
)

__all__ = [
    "BaseVLMBackend",
    "VLMOutput",
    "HazardConfig",
    "Phi4MultimodalBackend",
    "TinyLocalVLMBackend",
    "StubBackend",
    "HOIKey",
    "HazardOutput",
    "HazardReasoner",
    "HazardReasoningOutput",
    "DistilledHazardReasoner",
    "LaCHazardReasoner",
]
