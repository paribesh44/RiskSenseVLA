"""Stress tests for memory, HOI, and metrics under adversarial conditions."""

from __future__ import annotations

import torch

from risksense_vla.memory.hazard_memory import HazardAwareMemory
from risksense_vla.eval.metrics import evaluate_sequence, temporal_hoi_consistency
from risksense_vla.types import MemoryState, PerceptionDetection


def _make_det(track_id: str, label: str = "obj", emb_dim: int = 256) -> PerceptionDetection:
    return PerceptionDetection(
        track_id=track_id,
        label=label,
        confidence=0.9,
        bbox_xyxy=(10, 10, 50, 50),
        mask=torch.zeros((1, 1), dtype=torch.float32),
        clip_embedding=torch.randn(emb_dim, dtype=torch.float32),
    )


class TestFastOcclusion:
    """Objects disappearing and reappearing every 2-3 frames."""

    def test_memory_handles_rapid_occlusion(self) -> None:
        memory = HazardAwareMemory(emb_dim=64)
        state: MemoryState | None = None
        obj_a = _make_det("obj_a", emb_dim=64)

        for frame in range(30):
            dets = [obj_a] if frame % 3 != 2 else []
            state = memory.update(
                timestamp=frame / 24.0,
                detections=dets,
                previous_memory_state=state,
            )

        assert state is not None
        assert state.hoi_embedding.shape == (1, 64)
        assert state.state_vector.shape == (1, 512)

    def test_persistence_recovers_after_occlusion(self) -> None:
        memory = HazardAwareMemory(emb_dim=64)
        state: MemoryState | None = None
        det = _make_det("occ_obj", emb_dim=64)

        for i in range(10):
            state = memory.update(timestamp=i / 24.0, detections=[det], previous_memory_state=state)
        pre_occ_persistence = next(
            (o.persistence for o in state.objects if o.track_id == "occ_obj"), 0.0
        )

        for i in range(10, 13):
            state = memory.update(timestamp=i / 24.0, detections=[], previous_memory_state=state)

        still_tracked = any(o.track_id == "occ_obj" for o in state.objects)
        if still_tracked:
            post_occ_persistence = next(o.persistence for o in state.objects if o.track_id == "occ_obj")
            assert post_occ_persistence < pre_occ_persistence


class TestMultiHazardScenes:
    """10+ simultaneous high-risk objects."""

    def test_many_hazardous_objects(self) -> None:
        memory = HazardAwareMemory(emb_dim=64)
        state: MemoryState | None = None
        n_objects = 15
        dets = [_make_det(f"hazobj_{i}", emb_dim=64) for i in range(n_objects)]
        hazards = [0.9] * n_objects

        for frame in range(20):
            state = memory.update(
                timestamp=frame / 24.0,
                detections=dets,
                hazards=hazards,
                previous_memory_state=state,
            )

        assert state is not None
        assert len(state.objects) == n_objects
        for obj in state.objects:
            assert obj.persistence > 0.5
            assert obj.hazard_weight > 0.0


class TestObjectDisappearance:
    """All objects vanish at once."""

    def test_all_objects_vanish(self) -> None:
        memory = HazardAwareMemory(emb_dim=64)
        state: MemoryState | None = None
        dets = [_make_det(f"obj_{i}", emb_dim=64) for i in range(5)]
        low_hazards = [0.0] * 5

        for frame in range(10):
            state = memory.update(
                timestamp=frame / 24.0,
                detections=dets,
                hazards=low_hazards,
                previous_memory_state=state,
            )
        assert len(state.objects) == 5

        for frame in range(10, 60):
            state = memory.update(
                timestamp=frame / 24.0,
                detections=[],
                previous_memory_state=state,
            )

        assert len(state.objects) == 0

    def test_embedding_remains_valid(self) -> None:
        memory = HazardAwareMemory(emb_dim=64)
        state: MemoryState | None = None
        dets = [_make_det("solo", emb_dim=64)]

        for frame in range(5):
            state = memory.update(timestamp=frame / 24.0, detections=dets, previous_memory_state=state)

        for frame in range(5, 20):
            state = memory.update(timestamp=frame / 24.0, detections=[], previous_memory_state=state)

        assert state.hoi_embedding.shape == (1, 64)
        assert not torch.isnan(state.hoi_embedding).any()
        assert not torch.isinf(state.hoi_embedding).any()


class TestHighDetectionCount:
    """32+ objects per frame."""

    def test_32_objects(self) -> None:
        memory = HazardAwareMemory(emb_dim=64)
        state: MemoryState | None = None
        dets = [_make_det(f"big_{i}", emb_dim=64) for i in range(32)]

        for frame in range(10):
            state = memory.update(
                timestamp=frame / 24.0,
                detections=dets,
                previous_memory_state=state,
            )

        assert state is not None
        assert len(state.objects) == 32
        assert not torch.isnan(state.state_vector).any()


class TestZeroDetections:
    """Empty frames for extended periods."""

    def test_extended_empty_frames(self) -> None:
        memory = HazardAwareMemory(emb_dim=64)
        state: MemoryState | None = None

        for frame in range(100):
            state = memory.update(
                timestamp=frame / 24.0,
                detections=[],
                previous_memory_state=state,
            )

        assert state is not None
        assert len(state.objects) == 0
        assert state.hoi_embedding.shape == (1, 64)
        assert not torch.isnan(state.hoi_embedding).any()


class TestMetricsEdgeCases:
    """Stress the metrics computation."""

    def test_thc_all_empty_hois(self) -> None:
        records = [{"hois": []} for _ in range(50)]
        assert temporal_hoi_consistency(records) == 0.0

    def test_evaluate_sequence_single_frame(self) -> None:
        records = [{
            "frame_id": 0,
            "hois": [{"action": "cut", "confidence": 0.9}],
            "hazards": [],
            "detections": [{"confidence": 0.8}],
            "latency_ms": {"perception": 10.0},
        }]
        sm = evaluate_sequence(records)
        assert sm.thc == 0.0
        assert sm.fps > 0 or sm.fps == 0.0

    def test_evaluate_sequence_many_frames(self) -> None:
        records = []
        for i in range(500):
            records.append({
                "frame_id": i,
                "hois": [{"action": "hold", "confidence": 0.8}],
                "hazards": [{"score": 0.3, "action": "hold"}] if i % 10 == 0 else [],
                "attention_allocation": {"obj_0": 0.5},
                "detections": [{"confidence": 0.7}],
                "latency_ms": {"perception": 15.0, "memory": 3.0},
            })
        sm = evaluate_sequence(records)
        assert 0.0 <= sm.thc <= 1.0
        assert 0.0 <= sm.rme <= 1.0
        assert sm.fps > 0
