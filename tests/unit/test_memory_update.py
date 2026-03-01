from __future__ import annotations

import torch

from hapvla.memory import HazardAwareMemory
from hapvla.types import Detection, HazardScore


def test_hazard_weighted_persistence() -> None:
    mem = HazardAwareMemory()
    det = Detection(track_id="t1", label="knife", confidence=0.9, bbox_xyxy=(10, 10, 50, 50))
    emb = torch.randn(1, 256)
    low = [HazardScore("human", "hold", "knife", 0.1, "low", "low-risk")]
    high = [HazardScore("human", "cut", "knife", 0.9, "high", "high-risk")]

    s1 = mem.update(timestamp=0.0, detections=[det], embeddings=emb, hazards=low)
    p1 = s1.objects[0].persistence
    s2 = mem.update(timestamp=1.0, detections=[det], embeddings=emb, hazards=high)
    p2 = s2.objects[0].persistence
    assert p2 >= p1


def test_stale_objects_decay() -> None:
    mem = HazardAwareMemory()
    det = Detection(track_id="t2", label="box", confidence=0.8, bbox_xyxy=(0, 0, 20, 20))
    emb = torch.randn(1, 256)
    mem.update(timestamp=0.0, detections=[det], embeddings=emb, hazards=[])
    for i in range(30):
        mem.update(timestamp=float(i + 1), detections=[], embeddings=torch.zeros(0, 256), hazards=[])
    assert "t2" not in mem.objects
