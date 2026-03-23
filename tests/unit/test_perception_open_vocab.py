from __future__ import annotations

import numpy as np

from risksense_vla.perception import OpenVocabPerception


def _red_blob_frame(height: int = 240, width: int = 320) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[80:160, 100:220, 2] = 255
    return frame


def test_open_vocab_contract_shapes_and_indices() -> None:
    cfg = {
        "perception": {
            "detector_backend": "mock",
            "allow_mock_backend": True,
            "embedder_backend": "fallback",
            "embedding_dim": 256,
            "detector_max_detections": 6,
        }
    }
    perception = OpenVocabPerception.from_config(cfg=cfg, device="cpu")
    detections = perception.infer(_red_blob_frame(), labels=["knife", "stove"])

    assert len(detections) >= 1

    h, w = 240, 320
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        assert isinstance(x1, int) and isinstance(y1, int)
        assert isinstance(x2, int) and isinstance(y2, int)
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
        assert det.mask.ndim == 2
        assert det.clip_embedding.ndim == 1
        assert det.clip_embedding.shape[0] == 256
        assert det.label in {"knife", "stove"}


def test_open_vocab_empty_or_small_input_still_returns_valid_types() -> None:
    cfg = {
        "perception": {
            "detector_backend": "mock",
            "allow_mock_backend": True,
            "embedder_backend": "fallback",
            "embedding_dim": 256,
        }
    }
    perception = OpenVocabPerception.from_config(cfg=cfg, device="cpu")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = perception.infer(frame, labels=["person"])

    assert isinstance(detections, list)
    for det in detections:
        assert det.mask.ndim == 2
        assert det.clip_embedding.ndim == 1
        assert det.clip_embedding.shape[0] == 256
