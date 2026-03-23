from __future__ import annotations

import json

import torch

from risksense_vla.hoi import (
    PredictiveHOINet,
    PredictiveHOIModule,
    TemporalHOIPreprocessedDataset,
    build_hoi_dataloader,
    evaluate_predictive_hoi,
    train_predictive_hoi,
)
from risksense_vla.memory import HazardAwareMemory
from risksense_vla.types import MemoryState, PerceptionDetection


def _det(track_id: str, label: str, emb_dim: int = 256) -> PerceptionDetection:
    emb = torch.linspace(0.0, 1.0, steps=emb_dim, dtype=torch.float32)
    mask = torch.zeros((32, 32), dtype=torch.float32)
    mask[4:20, 4:20] = 1.0
    return PerceptionDetection(
        track_id=track_id,
        label=label,
        confidence=0.9,
        bbox_xyxy=(0, 0, 10, 10),
        mask=mask,
        clip_embedding=emb,
    )


def test_predictive_hoi_output_types_and_shapes() -> None:
    module = PredictiveHOIModule(future_horizon_seconds=3, emb_dim=256)
    memory = MemoryState(timestamp=0.0, hoi_embedding=torch.ones(1, 256), state_vector=torch.ones(1, 512))
    dets = [_det("1", "knife"), _det("2", "bottle")]
    out = module.infer(memory_state=memory, object_detections=dets, timestamp=3.0)

    assert len(out.hoi_current) == 2
    assert out.hoi_future_embeddings.shape == (2, 3, 256)
    assert out.future_action_confidences.shape == (2, 3)
    assert len(out.future_action_labels) == 2
    for hoi in out.hoi_current:
        assert isinstance(hoi.subject, str)
        assert isinstance(hoi.action, str)
        assert isinstance(hoi.object, str)
        assert 0.0 <= hoi.confidence <= 1.0
        assert abs(hoi.t_start - 3.0) < 1e-6
        assert abs(hoi.t_end - 3.0) < 1e-6


def test_predictive_hoi_empty_input_contract() -> None:
    module = PredictiveHOIModule(future_horizon_seconds=3, emb_dim=256)
    memory = MemoryState(timestamp=0.0)
    out = module.infer(memory_state=memory, object_detections=[], timestamp=1.0, horizon_seconds=2)
    assert out.hoi_current == []
    assert out.hoi_future_embeddings.shape == (0, 2, 256)
    assert out.future_action_confidences.shape == (0, 2)
    assert out.future_action_labels == []


def test_predictive_hoi_temporal_coherence() -> None:
    module = PredictiveHOIModule(future_horizon_seconds=3, emb_dim=256)
    memory = HazardAwareMemory(emb_dim=256)
    det = _det("same", "knife")

    actions = []
    futures = []
    for ts in (0.0, 1.0, 2.0):
        mem_state = memory.update(timestamp=ts, detections=[det], hazards=None)
        out = module.infer(memory_state=mem_state, object_detections=[det], timestamp=ts)
        actions.append(out.hoi_current[0].action)
        futures.append(out.hoi_future_embeddings[0, 0, :])

    assert actions[0] == actions[1] == actions[2]
    cos_01 = torch.nn.functional.cosine_similarity(futures[0].unsqueeze(0), futures[1].unsqueeze(0)).item()
    cos_12 = torch.nn.functional.cosine_similarity(futures[1].unsqueeze(0), futures[2].unsqueeze(0)).item()
    assert cos_01 > 0.0
    assert cos_12 > 0.0


def test_preprocessed_dataset_and_training_shapes(tmp_path) -> None:
    rec = {
        "video_id": "vid0",
        "start_frame": 0,
        "end_frame": 2,
        "hois": [
            {"subject": "human", "action": "hold", "object": "knife", "frame_idx": 0},
            {"subject": "human", "action": "cut", "object": "knife", "frame_idx": 1},
            {"subject": "human", "action": "drop", "object": "knife", "frame_idx": 2},
        ],
    }
    data_path = tmp_path / "hoi.jsonl"
    data_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    actions = ["hold", "cut", "drop"]
    dataset = TemporalHOIPreprocessedDataset(str(data_path), action_vocab=actions, emb_dim=256, horizon_seconds=3)
    loader = build_hoi_dataloader(dataset, batch_size=1, shuffle=False)

    model = PredictiveHOINet(emb_dim=256, num_actions=len(actions), horizon_seconds=3)
    history = train_predictive_hoi(model, loader, epochs=1, lr=1e-3, device="cpu", use_amp=False)
    metrics = evaluate_predictive_hoi(model, loader, device="cpu")

    assert "loss" in history and len(history["loss"]) == 1
    assert "current_top1" in metrics
    assert "future_embedding_cosine" in metrics
