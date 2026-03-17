from __future__ import annotations

import torch

from risksense_vla.hoi import PredictiveHOIModule
from risksense_vla.types import MemoryState, PerceptionDetection


def test_predictive_hoi_outputs_current_and_future_embeddings() -> None:
    predictor = PredictiveHOIModule(future_horizon_seconds=3, emb_dim=256)
    mask = torch.zeros((32, 32), dtype=torch.float32)
    mask[5:16, 5:16] = 1.0
    det = PerceptionDetection(
        track_id="1",
        label="knife",
        confidence=0.9,
        bbox_xyxy=(0, 0, 10, 10),
        mask=mask,
        clip_embedding=torch.randn(256),
    )
    memory = MemoryState(timestamp=0.0, hoi_embedding=torch.randn(1, 256), state_vector=torch.randn(1, 512))
    out = predictor.infer(memory_state=memory, object_detections=[det], timestamp=10.0)
    assert len(out.hoi_current) == 1
    assert out.hoi_future_embeddings.shape == (1, 3, 256)
