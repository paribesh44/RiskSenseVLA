from __future__ import annotations

import torch

from risksense_vla.hoi import ProtoHOIPredictor
from risksense_vla.types import Detection, MemoryState


def test_protohoi_outputs_current_and_future() -> None:
    predictor = ProtoHOIPredictor(future_horizon_seconds=3)
    det = Detection(track_id="1", label="knife", confidence=0.9, bbox_xyxy=(0, 0, 10, 10), embedding_idx=0)
    embeddings = torch.randn(1, 256)
    memory = MemoryState(timestamp=0.0, hoi_embedding=torch.randn(1, 256), state_vector=torch.randn(1, 512))
    hois = predictor.predict(timestamp=10.0, detections=[det], embeddings=embeddings, memory=memory)
    current = [h for h in hois if not h.predicted]
    future = [h for h in hois if h.predicted]
    assert len(current) == 1
    assert len(future) == 3
