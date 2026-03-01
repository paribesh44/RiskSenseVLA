from __future__ import annotations

import numpy as np

from risksense_vla.perception import BoxMaskSegmenter, CLIPEmbedder, FallbackEmbedder, OpenVocabPerception
from risksense_vla.perception.open_vocab import Detector
from risksense_vla.types import Detection


class BrokenDetector(Detector):
    def detect(self, frame_bgr: np.ndarray, open_vocab_labels: list[str] | None = None) -> list[Detection]:
        _ = frame_bgr, open_vocab_labels
        raise RuntimeError("forced detector failure")


def test_box_mask_segmenter_shape_contract() -> None:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    detections = [
        Detection(track_id="a", label="knife", confidence=0.9, bbox_xyxy=(10, 8, 40, 40), embedding_idx=0),
        Detection(track_id="b", label="stove", confidence=0.8, bbox_xyxy=(45, 10, 90, 50), embedding_idx=1),
    ]
    masks = BoxMaskSegmenter().segment(frame, detections)
    assert masks.shape == (2, 64, 96)
    assert float(masks[0].sum().item()) > 0.0
    assert float(masks[1].sum().item()) > 0.0


def test_clip_embedder_disabled_uses_fallback_path() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[20:60, 30:70, 1] = 200
    detections = [
        Detection(track_id="x", label="object", confidence=0.7, bbox_xyxy=(20, 20, 80, 70), embedding_idx=0)
    ]
    embedder = CLIPEmbedder(enabled=False, output_dim=256, allow_fallback=True)
    emb = embedder.encode(frame, detections)
    assert emb.shape == (1, 256)


def test_detector_failure_falls_back_to_mock() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[20:80, 50:120, 2] = 255
    perception = OpenVocabPerception(
        detector=BrokenDetector(),
        segmenter=BoxMaskSegmenter(),
        embedder=FallbackEmbedder(dim=256),
    )
    out = perception.infer(frame, labels=["knife"])
    assert len(out.detections) >= 1
    assert out.embeddings.shape[1] == 256
