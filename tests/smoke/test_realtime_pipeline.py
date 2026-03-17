from __future__ import annotations

import numpy as np
import torch

from risksense_vla.attention import SemanticAttentionScheduler
from risksense_vla.hazard import LaCHazardReasoner
from risksense_vla.hoi import PredictiveHOIModule, ProtoHOIPredictor
from risksense_vla.memory import HazardAwareMemory
from risksense_vla.perception import OpenVocabPerception


def test_smoke_one_frame_pipeline() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:140, 120:200, 2] = 255  # red blob for fallback detector
    perception = OpenVocabPerception.from_config(
        cfg={
            "perception": {
                "detector_backend": "mock",
                "allow_mock_backend": True,
                "embedder_backend": "fallback",
            }
        }
    )
    memory = HazardAwareMemory()
    hoi = ProtoHOIPredictor()
    reasoner = LaCHazardReasoner(backend_type="stub", lightweight_mode=True)
    attention = SemanticAttentionScheduler()

    detections = perception.infer(frame)
    mem0 = memory.update(timestamp=0.0, detections=detections, hazards=None)
    hois = hoi.predict(timestamp=0.0, detections=detections, memory=mem0)
    hazard_out = reasoner.predict_hazard(
        hoi_current=hois,
        hoi_future_embeddings=torch.zeros((len(hois), 3, 256), dtype=torch.float32),
        memory_state=mem0,
        frame_bgr=frame,
    )
    allocation = attention.allocation(detections, hazard_out.hazards)

    assert len(detections) >= 1
    assert len(hois) >= 1
    assert isinstance(hazard_out.hazard_map, dict)
    assert len(hazard_out.hazard_map) >= 1
    assert all(0.0 <= h.score <= 1.0 for h in hazard_out.hazards)
    assert all(0.0 <= s <= 1.0 for s in hazard_out.hazard_map.values())
    mem_ids = {obj.track_id for obj in mem0.objects}
    allowed_unknown = {key for key in hazard_out.hazard_map if key.startswith("unknown:")}
    assert set(hazard_out.hazard_map).issubset(mem_ids | allowed_unknown)
    assert len(hazard_out.hazard_map_legacy) >= 1
    assert isinstance(allocation, dict)


def test_smoke_predictive_hoi_infer() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:140, 120:200, 2] = 255
    perception = OpenVocabPerception.from_config(
        cfg={
            "perception": {
                "detector_backend": "mock",
                "allow_mock_backend": True,
                "embedder_backend": "fallback",
            }
        }
    )
    memory = HazardAwareMemory()
    predictor = PredictiveHOIModule(future_horizon_seconds=3)

    detections = perception.infer(frame)
    mem = memory.update(timestamp=0.0, detections=detections, hazards=None)
    hoi_out = predictor.infer(memory_state=mem, object_detections=detections, timestamp=0.0)

    assert len(hoi_out.hoi_current) >= 1
    assert hoi_out.hoi_future_embeddings.shape[0] == len(detections)
    assert hoi_out.hoi_future_embeddings.shape[1] == 3
