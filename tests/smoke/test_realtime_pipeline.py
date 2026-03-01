from __future__ import annotations

import numpy as np

from hapvla.attention import SemanticAttentionScheduler
from hapvla.hazard import LaCHazardReasoner
from hapvla.hoi import ProtoHOIPredictor
from hapvla.memory import HazardAwareMemory
from hapvla.perception import OpenVocabPerception


def test_smoke_one_frame_pipeline() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:140, 120:200, 2] = 255  # red blob for fallback detector
    perception = OpenVocabPerception.default()
    memory = HazardAwareMemory()
    hoi = ProtoHOIPredictor()
    reasoner = LaCHazardReasoner()
    attention = SemanticAttentionScheduler()

    out = perception.infer(frame)
    mem0 = memory.update(timestamp=0.0, detections=out.detections, embeddings=out.embeddings, hazards=[])
    hois = hoi.predict(timestamp=0.0, detections=out.detections, embeddings=out.embeddings, memory=mem0)
    hazard_out = reasoner.infer(hois)
    allocation = attention.allocation(out.detections, hazard_out.hazards)

    assert len(out.detections) >= 1
    assert len(hois) >= 1
    assert isinstance(allocation, dict)
