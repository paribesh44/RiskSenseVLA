from __future__ import annotations

import torch

from risksense_vla.memory import HazardAwareMemory, update_hazard_memory
from risksense_vla.memory.hazard_memory import update_state
from risksense_vla.types import HazardScore, MemoryState, PerceptionDetection


def _emb(count: int, dim: int = 256) -> torch.Tensor:
    if count == 0:
        return torch.zeros((0, dim), dtype=torch.float32)
    return torch.linspace(0.0, 1.0, steps=count * dim, dtype=torch.float32).reshape(count, dim)


def _det(
    *,
    track_id: str,
    label: str,
    confidence: float,
    bbox_xyxy: tuple[int, int, int, int],
    emb: torch.Tensor,
) -> PerceptionDetection:
    mask = torch.zeros((128, 128), dtype=torch.float32)
    x1, y1, x2, y2 = bbox_xyxy
    mask[y1:y2, x1:x2] = 1.0
    return PerceptionDetection(
        track_id=track_id,
        label=label,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
        mask=mask,
        clip_embedding=emb.detach().clone().to(torch.float32),
    )


def _obj_persistence(memory_state: MemoryState) -> dict[str, float]:
    return {obj.track_id: obj.persistence for obj in memory_state.objects}


def test_hazard_weighted_persistence_prefers_high_risk() -> None:
    mem = HazardAwareMemory()
    emb = _emb(2)
    det_low = _det(track_id="low", label="box", confidence=0.9, bbox_xyxy=(0, 0, 40, 40), emb=emb[0])
    det_high = _det(track_id="high", label="knife", confidence=0.9, bbox_xyxy=(50, 50, 100, 100), emb=emb[1])
    mem.update(
        timestamp=0.0,
        detections=[det_low, det_high],
        hazards=[0.05, 0.95],
    )

    for i in range(30):
        mem.update(timestamp=float(i + 1), detections=[], hazards=None)

    assert "high" in mem.objects
    if "low" in mem.objects:
        assert mem.objects["high"].persistence > mem.objects["low"].persistence


def test_dynamic_entry_exit_and_decay() -> None:
    mem = HazardAwareMemory()
    emb0 = _emb(2)
    emb1 = _emb(2)
    det_a = _det(track_id="a", label="box", confidence=0.7, bbox_xyxy=(0, 0, 20, 20), emb=emb0[0])
    det_b0 = _det(track_id="b", label="glass", confidence=0.8, bbox_xyxy=(10, 10, 40, 40), emb=emb0[1])
    det_b1 = _det(track_id="b", label="glass", confidence=0.8, bbox_xyxy=(12, 12, 44, 44), emb=emb1[0])
    det_c = _det(track_id="c", label="knife", confidence=0.95, bbox_xyxy=(30, 30, 70, 70), emb=emb1[1])

    mem.update(timestamp=0.0, detections=[det_a, det_b0], hazards=[0.0, 0.0])
    state = mem.update(timestamp=1.0, detections=[det_b1, det_c], hazards=[0.0, 1.0])

    ids = {obj.track_id for obj in state.objects}
    assert {"a", "b", "c"}.issubset(ids)
    assert mem.objects["b"].age_frames == 2
    assert mem.objects["b"].last_bbox_xyxy == (12, 12, 44, 44)

    for i in range(34):
        mem.update(timestamp=float(i + 2), detections=[], hazards=None)

    assert "a" not in mem.objects
    assert "c" in mem.objects


def test_functional_api_matches_class_api() -> None:
    hazard_events = [HazardScore("human", "hold", "knife", 0.8, "high", "high-risk")]
    steps = [
        (
            0.0,
            [_det(track_id="x", label="knife", confidence=0.9, bbox_xyxy=(1, 1, 11, 11), emb=_emb(1)[0])],
            [0.8],
            hazard_events,
        ),
        (
            1.0,
            [_det(track_id="x", label="knife", confidence=0.9, bbox_xyxy=(2, 2, 12, 12), emb=_emb(1)[0])],
            [0.8],
            hazard_events,
        ),
        (2.0, [], None, []),
    ]

    class_mem = HazardAwareMemory()
    prev_state = None
    for timestamp, detections, hazards, events in steps:
        class_state = class_mem.update(
            timestamp=timestamp,
            detections=detections,
            hazards=hazards,
            hazard_events=events,
        )
        func_state = update_hazard_memory(
            timestamp=timestamp,
            detections=detections,
            previous_memory_state=prev_state,
            hazards=hazards,
            hazard_events=events,
        )
        prev_state = func_state

        assert class_state.hoi_embedding.shape == func_state.hoi_embedding.shape == (1, 256)
        assert class_state.state_vector.shape == func_state.state_vector.shape == (1, 512)
        assert _obj_persistence(class_state) == _obj_persistence(func_state)
        assert torch.allclose(class_state.hoi_embedding, func_state.hoi_embedding, atol=1e-6)


def test_state_shapes_and_optional_logging_callback() -> None:
    mem = HazardAwareMemory()
    logs: list[dict[str, float]] = []
    state = mem.update(
        timestamp=3.0,
        detections=[],
        hazards=None,
        log_callback=lambda payload: logs.append(payload),
    )

    assert state.hoi_embedding.shape == (1, 256)
    assert state.state_vector.shape == (1, 512)
    assert torch.isfinite(state.hoi_embedding).all()
    assert torch.isfinite(state.state_vector).all()
    assert abs(state.state_vector[0, 0].item() - 0.0) < 1e-6
    assert len(logs) == 1
    assert abs(logs[0]["timestamp"] - 3.0) < 1e-6


def test_memory_contract_alignment() -> None:
    mem = HazardAwareMemory()
    detections = [
        _det(
            track_id="trk1",
            label="knife",
            confidence=0.91,
            bbox_xyxy=(8, 8, 24, 24),
            emb=_emb(1)[0],
        )
    ]
    state = mem.update(timestamp=0.0, detections=detections, hazards=[0.8])
    assert len(state.objects) == 1
    assert state.objects[0].track_id == "trk1"
    assert state.hoi_embedding.shape == (1, 256)


def test_update_state_high_hazard_decays_slower_when_unseen() -> None:
    low_obj = _det(track_id="low", label="box", confidence=0.8, bbox_xyxy=(0, 0, 20, 20), emb=_emb(1)[0])
    high_obj = _det(track_id="high", label="knife", confidence=0.8, bbox_xyxy=(20, 20, 40, 40), emb=_emb(1)[0])
    mem = HazardAwareMemory()
    mem.update(timestamp=0.0, detections=[low_obj, high_obj], hazards=[0.05, 0.95])
    low_before = mem.objects["low"].persistence
    high_before = mem.objects["high"].persistence

    updated, observed_count, stale_count = update_state(
        prev_state=mem.objects,
        detections=[],
        hazard_scores=[],
        base_persistence=mem.base_persistence,
        base_decay=mem.base_decay,
        alpha=mem.alpha,
        beta=mem.beta,
        observation_boost=mem.observation_boost,
        min_persistence=mem.min_persistence,
        max_persistence=mem.max_persistence,
    )
    assert observed_count == 0
    assert stale_count == 2
    assert "low" in updated and "high" in updated
    low_drop = low_before - updated["low"].persistence
    high_drop = high_before - updated["high"].persistence
    assert high_drop < low_drop


def test_missing_hazards_default_to_zero_not_one() -> None:
    mem = HazardAwareMemory()
    det = _det(track_id="x", label="knife", confidence=0.9, bbox_xyxy=(2, 2, 10, 10), emb=_emb(1)[0])
    state = mem.update(timestamp=0.0, detections=[det], hazards=None, hazard_events=None)
    assert state.objects[0].hazard_weight == 0.0


def test_use_hazard_weighting_false_gates_all_terms() -> None:
    det_low = _det(track_id="low", label="box", confidence=0.8, bbox_xyxy=(0, 0, 20, 20), emb=_emb(1)[0])
    det_high = _det(track_id="high", label="knife", confidence=0.8, bbox_xyxy=(20, 20, 40, 40), emb=_emb(1)[0])
    mem = HazardAwareMemory(use_hazard_weighting=False, alpha=0.0, beta=0.0)
    state = mem.update(timestamp=0.0, detections=[det_low, det_high], hazards=[0.0, 1.0])
    pers = {obj.track_id: obj.persistence for obj in state.objects}
    hz = {obj.track_id: obj.hazard_weight for obj in state.objects}
    assert abs(pers["low"] - pers["high"]) < 1e-6
    assert hz["low"] == 0.0 and hz["high"] == 0.0
