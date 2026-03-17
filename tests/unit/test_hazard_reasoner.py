from __future__ import annotations

import re
import torch

from risksense_vla.hazard import (
    BaseVLMBackend,
    DistilledHazardReasoner,
    HazardConfig,
    HazardReasoner,
    VLMOutput,
)
from risksense_vla.hoi import HOI
from risksense_vla.types import HOITriplet, MemoryObjectState, MemoryState


def _memory_state() -> MemoryState:
    return MemoryState(
        timestamp=0.0,
        objects=[
            MemoryObjectState(
                track_id="trk_knife_01",
                label="knife",
                last_bbox_xyxy=(10, 10, 20, 20),
                persistence=0.85,
                hazard_weight=0.90,
                age_frames=12,
            ),
            MemoryObjectState(
                track_id="trk_bottle_01",
                label="bottle",
                last_bbox_xyxy=(40, 30, 60, 80),
                persistence=0.65,
                hazard_weight=0.10,
                age_frames=12,
            ),
        ],
        hoi_embedding=torch.ones((1, 256), dtype=torch.float32),
        state_vector=torch.zeros((1, 512), dtype=torch.float32),
    )


def _future(n: int) -> torch.Tensor:
    return torch.zeros((n, 3, 256), dtype=torch.float32)


def test_predict_hazard_dual_maps_and_alerts() -> None:
    reasoner = DistilledHazardReasoner(
        checkpoint_path=None,
        fallback_mode="prior_only",
        alert_threshold=0.60,
        backend_type="stub",
        lightweight_mode=True,
        debug_prompt=True,
    )
    hois = [
        HOITriplet(
            subject="human",
            action="cut",
            object="knife",
            confidence=0.92,
            t_start=0.0,
            t_end=0.0,
            predicted=False,
        )
    ]
    out = reasoner.predict_hazard(hoi_current=hois, hoi_future_embeddings=_future(len(hois)), memory_state=_memory_state())

    assert len(out.hazards) == 1
    assert "trk_knife_01" in out.hazard_map
    assert "human:cut:knife" in out.hazard_map_legacy
    assert 0.0 <= out.hazards[0].score <= 1.0
    assert out.hazard_alerts == out.alerts
    assert out.explanations["trk_knife_01"]
    assert "human:cut:knife" in out.prompt_debug
    assert isinstance(out.alerts, list)


def test_predict_hazard_empty_input_contract() -> None:
    reasoner = DistilledHazardReasoner(checkpoint_path=None, backend_type="stub", lightweight_mode=True)
    out = reasoner.predict_hazard(hoi_current=[], hoi_future_embeddings=_future(0), memory_state=_memory_state())
    assert out.hazards == []
    assert out.alerts == []
    assert out.hazard_map == {}
    assert out.hazard_map_legacy == {}


def test_predict_hazard_accepts_phase3_hoi_dataclass() -> None:
    reasoner = DistilledHazardReasoner(
        checkpoint_path=None,
        fallback_mode="prior_only",
        backend_type="stub",
        lightweight_mode=True,
    )
    hoi = HOI(subject="human", action="carry", object="bottle", confidence=0.7, t_start=1.0, t_end=1.0)
    out = reasoner.predict_hazard(hoi_current=[hoi], hoi_future_embeddings=_future(1), memory_state=_memory_state())
    assert len(out.hazards) == 1
    assert "trk_bottle_01" in out.hazard_map


def test_infer_compatibility_shim() -> None:
    reasoner = DistilledHazardReasoner(checkpoint_path=None, backend_type="stub", lightweight_mode=True)
    out = reasoner.infer(
        [
            HOITriplet(
                subject="human",
                action="touch_hot_surface",
                object="stove",
                confidence=0.8,
                t_start=0.0,
                t_end=0.0,
                predicted=False,
            )
        ]
    )
    assert len(out.hazards) == 1
    assert len(out.hazard_map) == 1
    only_key = next(iter(out.hazard_map))
    assert only_key.startswith("unknown:")


def test_deterministic_fallback_when_checkpoint_missing() -> None:
    reasoner = DistilledHazardReasoner(
        checkpoint_path="artifacts/does_not_exist.pt",
        fallback_mode="blend",
        alert_threshold=0.4,
        backend_type="tiny",
        lightweight_mode=True,
    )
    hois = [
        HOITriplet(
            subject="human",
            action="touch_hot_surface",
            object="stove",
            confidence=0.7,
            t_start=2.0,
            t_end=2.0,
            predicted=True,
        )
    ]
    memory = _memory_state()
    out_a = reasoner.predict_hazard(hoi_current=hois, hoi_future_embeddings=_future(len(hois)), memory_state=memory)
    out_b = reasoner.predict_hazard(hoi_current=hois, hoi_future_embeddings=_future(len(hois)), memory_state=memory)

    assert out_a.hazard_map == out_b.hazard_map
    assert out_a.hazard_map_legacy == out_b.hazard_map_legacy
    assert [h.score for h in out_a.hazards] == [h.score for h in out_b.hazards]
    assert out_a.alerts == out_b.alerts


def test_score_bounds_for_all_outputs() -> None:
    reasoner = DistilledHazardReasoner(checkpoint_path=None, backend_type="stub", lightweight_mode=True)
    hois = [
        HOITriplet("human", "cut", "knife", confidence=3.5, t_start=0.0, t_end=0.0, predicted=False),
        HOITriplet("human", "carry", "bottle", confidence=-2.0, t_start=0.0, t_end=0.0, predicted=True),
    ]
    out = reasoner.predict_hazard(hoi_current=hois, hoi_future_embeddings=_future(len(hois)), memory_state=_memory_state())
    assert all(0.0 <= h.score <= 1.0 for h in out.hazards)
    assert all(0.0 <= score <= 1.0 for score in out.hazard_map.values())


def test_track_id_mapping_prefers_best_memory_object() -> None:
    memory = MemoryState(
        timestamp=0.0,
        objects=[
            MemoryObjectState(
                track_id="trk_knife_low",
                label="knife",
                last_bbox_xyxy=(10, 10, 20, 20),
                persistence=0.4,
                hazard_weight=0.2,
                age_frames=3,
            ),
            MemoryObjectState(
                track_id="trk_knife_best",
                label="knife",
                last_bbox_xyxy=(15, 15, 35, 35),
                persistence=0.9,
                hazard_weight=0.8,
                age_frames=9,
            ),
        ],
        hoi_embedding=torch.ones((1, 256), dtype=torch.float32),
        state_vector=torch.zeros((1, 512), dtype=torch.float32),
    )
    reasoner = DistilledHazardReasoner(checkpoint_path=None, backend_type="stub", lightweight_mode=True)
    out = reasoner.predict_hazard(
        hoi_current=[HOITriplet("human", "cut", "knife", confidence=0.8, t_start=0.0, t_end=0.0)],
        hoi_future_embeddings=_future(1),
        memory_state=memory,
    )
    assert "trk_knife_best" in out.hazard_map
    assert "trk_knife_low" not in out.hazard_map


def test_backend_swap_tiny_vs_stub() -> None:
    hois = [HOITriplet("human", "cut", "knife", confidence=0.8, t_start=0.0, t_end=0.0)]
    memory = _memory_state()
    stub_reasoner = DistilledHazardReasoner(checkpoint_path=None, backend_type="stub", lightweight_mode=True)
    tiny_reasoner = DistilledHazardReasoner(checkpoint_path=None, backend_type="tiny", lightweight_mode=True)
    out_stub = stub_reasoner.predict_hazard(hoi_current=hois, hoi_future_embeddings=_future(len(hois)), memory_state=memory)
    out_tiny = tiny_reasoner.predict_hazard(hoi_current=hois, hoi_future_embeddings=_future(len(hois)), memory_state=memory)
    assert out_stub.backend == "stub"
    assert out_tiny.backend == "tiny"
    assert all(0.0 <= s <= 1.0 for s in out_stub.hazard_map.values())
    assert all(0.0 <= s <= 1.0 for s in out_tiny.hazard_map.values())


class _FakeBackend(BaseVLMBackend):
    def predict_risk(
        self,
        prompt: str,
        image,
        hoi_embedding=None,
        future_embedding=None,
        memory_embedding=None,
    ):  # type: ignore[override]
        del prompt, image, hoi_embedding, future_embedding, memory_embedding
        return VLMOutput(generated_text="Risk score: 1.7\nExplanation: parsed explanation.")


def test_explanation_parsing_and_clamping() -> None:
    reasoner = HazardReasoner(
        backend=_FakeBackend(),
        config=HazardConfig(alert_threshold=0.6, backend_type="stub", debug_prompt=False),
    )
    out = reasoner.predict_hazard(
        hoi_current=[HOITriplet("human", "touch_hot_surface", "stove", 0.8, 0.0, 0.0)],
        hoi_future_embeddings=_future(1),
        memory_state=_memory_state(),
    )
    assert len(out.hazards) == 1
    assert abs(out.hazards[0].score - 1.0) < 1e-6
    assert re.search(r"parsed explanation", out.hazards[0].explanation, flags=re.IGNORECASE)

