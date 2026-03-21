from __future__ import annotations

import random

import torch

from scripts.benchmark_phase4 import _apply_occlusion_with_base_mask
from risksense_vla.experimental import (
    apply_occlusion,
    get_bool,
    resolve_mode,
    seed_everything,
    top_predicted_actions_by_horizon,
)
from risksense_vla.types import HOITriplet, PerceptionDetection


def _det(track_id: str) -> PerceptionDetection:
    return PerceptionDetection(
        track_id=track_id,
        label="obj",
        confidence=0.9,
        bbox_xyxy=(1, 1, 10, 10),
        mask=torch.zeros((1, 1), dtype=torch.float32),
        clip_embedding=torch.zeros((256,), dtype=torch.float32),
    )


def test_apply_occlusion_drops_all_when_probability_one() -> None:
    kept, events = apply_occlusion([_det("a"), _det("b")], occlusion_prob=1.0, rng=random.Random(1))
    assert kept == []
    assert len(events) == 2


def test_toggle_and_mode_resolution_precedence() -> None:
    cfg = {"memory": {"use_hazard_weighting": False}, "baselines": {"memory_mode": "naive"}}
    assert get_bool(cfg, "memory", "use_hazard_weighting", True) is False
    assert resolve_mode(cfg, "memory_mode", "hazard_aware") == "naive"


def test_seed_everything_deterministic() -> None:
    seed_everything(11)
    a = torch.randn(4)
    seed_everything(11)
    b = torch.randn(4)
    assert torch.allclose(a, b)


def test_top_predicted_actions_by_horizon() -> None:
    hois = [
        HOITriplet("human", "cut", "knife", 0.8, t_start=0.0, t_end=1.0, predicted=True),
        HOITriplet("human", "cut", "knife", 0.7, t_start=0.0, t_end=1.0, predicted=True),
        HOITriplet("human", "hold", "knife", 0.7, t_start=0.0, t_end=2.0, predicted=True),
    ]
    out = top_predicted_actions_by_horizon(hois, max_horizon=3)
    assert out[1] == "cut"
    assert out[2] == "hold"


def test_occlusion_levels_reuse_same_base_pattern_with_same_seed() -> None:
    detections = [_det("a"), _det("b"), _det("c"), _det("d")]
    base_mask: dict[str, float] = {}
    rng = random.Random(42)
    kept_low, _ = _apply_occlusion_with_base_mask(
        detections,
        occlusion_prob=0.25,
        frame_index=0,
        base_mask=base_mask,
        rng=rng,
    )
    kept_high, _ = _apply_occlusion_with_base_mask(
        detections,
        occlusion_prob=0.75,
        frame_index=0,
        base_mask=base_mask,
        rng=rng,
    )
    low_ids = {d.track_id for d in kept_low}
    high_ids = {d.track_id for d in kept_high}
    assert high_ids.issubset(low_ids)
