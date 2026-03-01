"""Open-vocabulary detection pipeline with pluggable detector backends."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np
import torch

from risksense_vla.perception.embed import CLIPEmbedder, FallbackEmbedder
from risksense_vla.perception.segment import BoxMaskSegmenter
from risksense_vla.types import Detection


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray, open_vocab_labels: list[str] | None = None) -> list[Detection]:
        ...


class Embedder(Protocol):
    def encode(self, frame_bgr: np.ndarray, detections: list[Detection]) -> torch.Tensor:
        ...


def _hash_track_id(label: str, bbox_xyxy: tuple[int, int, int, int]) -> str:
    raw = f"{label}:{bbox_xyxy[0]}:{bbox_xyxy[1]}:{bbox_xyxy[2]}:{bbox_xyxy[3]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def _normalize_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float] | list[float] | tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(x1, max(0, width - 1)))
    y1 = max(0, min(y1, max(0, height - 1)))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _prompt_labels(open_vocab_labels: list[str] | None, default_labels: list[str]) -> list[str]:
    labels = open_vocab_labels or default_labels
    cleaned = [x.strip() for x in labels if x and x.strip()]
    return cleaned if cleaned else list(default_labels)


@dataclass(slots=True)
class MockOpenVocabDetector:
    """Fast fallback detector that enables end-to-end real-time pipeline testing."""

    default_label: str = "person"
    max_detections: int = 4

    def detect(self, frame_bgr: np.ndarray, open_vocab_labels: list[str] | None = None) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        labels = _prompt_labels(open_vocab_labels, [self.default_label, "knife", "stove", "vehicle"])
        # Detect high-red regions as a proxy for potentially risky object-like blobs.
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 70, 40), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 40), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for i, cnt in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[: self.max_detections]):
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < 800:
                continue
            label = labels[(i + 1) % len(labels)]
            bbox = (x, y, x + bw, y + bh)
            detections.append(
                Detection(
                    track_id=_hash_track_id(label, bbox),
                    label=label,
                    confidence=0.55,
                    bbox_xyxy=bbox,
                    embedding_idx=i,
                )
            )
        if not detections:
            # Ensure at least one detection in fallback mode for downstream module smoke testing.
            x1, y1, x2, y2 = w // 3, h // 4, (2 * w) // 3, (3 * h) // 4
            detections = [
                Detection(
                    track_id="fallback0",
                    label=labels[0],
                    confidence=0.5,
                    bbox_xyxy=(x1, y1, x2, y2),
                    embedding_idx=0,
                )
            ]
        return detections


@dataclass(slots=True)
class GroundingDINOAdapter:
    """Grounding DINO adapter via Hugging Face Transformers."""

    model_name: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    max_detections: int = 12
    default_labels: list[str] = field(default_factory=lambda: ["person", "knife", "stove", "vehicle"])
    device: str = "cpu"
    local_files_only: bool = False
    _model: Any = None
    _processor: Any = None
    _init_failed: bool = False
    _init_error: str = ""

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if self._init_failed:
            raise RuntimeError(self._init_error or "Grounding DINO initialization previously failed.")
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            )
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            )
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:
            self._init_failed = True
            self._init_error = str(exc)
            raise

    def detect(self, frame_bgr: np.ndarray, open_vocab_labels: list[str] | None = None) -> list[Detection]:
        self._ensure_model()
        assert self._model is not None and self._processor is not None
        h, w = frame_bgr.shape[:2]
        labels = _prompt_labels(open_vocab_labels, self.default_labels)
        text_labels = [f"{label}." for label in labels]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=frame_rgb, text=text_labels, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            outputs = self._model(**model_inputs)

        result = None
        if hasattr(self._processor, "post_process_grounded_object_detection"):
            result = self._processor.post_process_grounded_object_detection(
                outputs,
                model_inputs.get("input_ids"),
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )
        elif hasattr(self._processor, "post_process_object_detection"):
            result = self._processor.post_process_object_detection(
                outputs, threshold=self.box_threshold, target_sizes=[(h, w)]
            )
        else:
            raise RuntimeError("Grounding DINO processor does not support post-processing for detections.")

        rows = result[0] if result else {}
        boxes = rows.get("boxes", [])
        scores = rows.get("scores", [])
        raw_labels = rows.get("labels", [])
        label_filter = set(labels)
        detections: list[Detection] = []
        for i in range(min(len(boxes), len(scores))):
            box = boxes[i].detach().cpu().tolist() if torch.is_tensor(boxes[i]) else boxes[i]
            normalized = _normalize_bbox_xyxy(box, width=w, height=h)
            if normalized is None:
                continue
            score = float(scores[i].item() if torch.is_tensor(scores[i]) else scores[i])
            if score < self.box_threshold:
                continue
            raw_label = raw_labels[i] if i < len(raw_labels) else "object"
            if torch.is_tensor(raw_label):
                raw_label = int(raw_label.item())
            if isinstance(raw_label, str):
                label = raw_label.strip().rstrip(".")
            elif isinstance(raw_label, int) and 0 <= raw_label < len(labels):
                label = labels[raw_label]
            else:
                label = str(raw_label)
            if label_filter and label not in label_filter:
                continue
            detections.append(
                Detection(
                    track_id=_hash_track_id(label, normalized),
                    label=label,
                    confidence=score,
                    bbox_xyxy=normalized,
                    embedding_idx=len(detections),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]


@dataclass(slots=True)
class YOLOE26Adapter:
    """YOLOE-26 style adapter via Ultralytics runtime."""

    model_path: str = "yoloe-26s.pt"
    confidence_threshold: float = 0.35
    max_detections: int = 12
    default_labels: list[str] = field(default_factory=lambda: ["person", "knife", "stove", "vehicle"])
    device: str = "cpu"
    _model: Any = None
    _last_prompt: tuple[str, ...] = field(default_factory=tuple)
    _init_failed: bool = False
    _init_error: str = ""

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._init_failed:
            raise RuntimeError(self._init_error or "YOLOE initialization previously failed.")
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        except Exception as exc:
            self._init_failed = True
            self._init_error = str(exc)
            raise

    def detect(self, frame_bgr: np.ndarray, open_vocab_labels: list[str] | None = None) -> list[Detection]:
        self._ensure_model()
        assert self._model is not None
        h, w = frame_bgr.shape[:2]
        labels = _prompt_labels(open_vocab_labels, self.default_labels)
        prompt = tuple(labels)
        if prompt != self._last_prompt and hasattr(self._model, "set_classes"):
            self._model.set_classes(labels)
            self._last_prompt = prompt

        results = self._model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []
        names = getattr(result, "names", getattr(self._model, "names", {}))
        label_filter = set(labels)
        detections: list[Detection] = []
        count = int(boxes.xyxy.shape[0])
        for i in range(count):
            xyxy = boxes.xyxy[i].detach().cpu().tolist()
            normalized = _normalize_bbox_xyxy(xyxy, width=w, height=h)
            if normalized is None:
                continue
            conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            if conf < self.confidence_threshold:
                continue
            cls_idx = int(boxes.cls[i].item()) if boxes.cls is not None else -1
            if isinstance(names, dict):
                label = str(names.get(cls_idx, cls_idx))
            elif isinstance(names, list) and 0 <= cls_idx < len(names):
                label = str(names[cls_idx])
            else:
                label = str(cls_idx)
            if label_filter and label not in label_filter:
                continue
            detections.append(
                Detection(
                    track_id=_hash_track_id(label, normalized),
                    label=label,
                    confidence=conf,
                    bbox_xyxy=normalized,
                    embedding_idx=len(detections),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]


@dataclass(slots=True)
class PerceptionOutput:
    detections: list[Detection]
    masks: torch.Tensor
    embeddings: torch.Tensor


@dataclass(slots=True)
class OpenVocabPerception:
    detector: Detector
    segmenter: BoxMaskSegmenter
    embedder: Embedder
    default_labels: list[str] = field(default_factory=lambda: ["person", "knife", "stove", "vehicle"])
    max_detections: int = 12
    fallback_detector: Detector = field(default_factory=MockOpenVocabDetector)

    @classmethod
    def default(cls) -> "OpenVocabPerception":
        return cls.from_config(cfg={})

    @classmethod
    def from_config(cls, cfg: dict[str, Any], device: str = "cpu") -> "OpenVocabPerception":
        p_cfg = cfg.get("perception", {}) if isinstance(cfg, dict) else {}
        m_cfg = cfg.get("models", {}) if isinstance(cfg, dict) else {}
        detector_name = str(p_cfg.get("detector_backend", m_cfg.get("detector", "mock"))).lower()
        embedder_name = str(p_cfg.get("embedder_backend", m_cfg.get("embedder", "fallback"))).lower()
        max_detections = int(p_cfg.get("detector_max_detections", 12))
        labels = p_cfg.get("default_labels", ["person", "knife", "stove", "vehicle"])
        labels = [str(x) for x in labels]
        if detector_name in {"grounding_dino", "groundingdino"}:
            detector: Detector = GroundingDINOAdapter(
                model_name=str(p_cfg.get("grounding_dino_model_id", "IDEA-Research/grounding-dino-base")),
                box_threshold=float(p_cfg.get("detector_confidence_threshold", 0.35)),
                text_threshold=float(p_cfg.get("detector_text_threshold", 0.25)),
                max_detections=max_detections,
                default_labels=labels,
                device=device,
                local_files_only=bool(p_cfg.get("local_files_only", False)),
            )
        elif detector_name in {"yoloe26", "yoloe-26", "yoloe"}:
            detector = YOLOE26Adapter(
                model_path=str(p_cfg.get("yoloe_model_path", "yoloe-26s.pt")),
                confidence_threshold=float(p_cfg.get("detector_confidence_threshold", 0.35)),
                max_detections=max_detections,
                default_labels=labels,
                device=device,
            )
        else:
            detector = MockOpenVocabDetector(default_label=labels[0] if labels else "person")

        embedding_dim = int(p_cfg.get("embedding_dim", 256))
        if embedder_name == "clip":
            embedder: Embedder = CLIPEmbedder(
                model_name=str(p_cfg.get("clip_model_id", "openai/clip-vit-base-patch32")),
                output_dim=embedding_dim,
                device=device,
                batch_size=int(p_cfg.get("clip_batch_size", 8)),
                enabled=True,
                allow_fallback=False,
                local_files_only=bool(p_cfg.get("local_files_only", False)),
            )
        elif embedder_name in {"clip_or_fallback", "clip-fallback"}:
            embedder = CLIPEmbedder(
                model_name=str(p_cfg.get("clip_model_id", "openai/clip-vit-base-patch32")),
                output_dim=embedding_dim,
                device=device,
                batch_size=int(p_cfg.get("clip_batch_size", 8)),
                enabled=True,
                allow_fallback=True,
                local_files_only=bool(p_cfg.get("local_files_only", False)),
            )
        else:
            embedder = FallbackEmbedder(dim=embedding_dim)

        return cls(
            detector=detector,
            segmenter=BoxMaskSegmenter(),
            embedder=embedder,
            default_labels=labels,
            max_detections=max_detections,
            fallback_detector=MockOpenVocabDetector(default_label=labels[0] if labels else "person"),
        )

    def infer(self, frame_bgr: np.ndarray, labels: list[str] | None = None) -> PerceptionOutput:
        active_labels = labels if labels is not None else self.default_labels
        try:
            detections = self.detector.detect(frame_bgr, open_vocab_labels=active_labels)
        except Exception:
            detections = self.fallback_detector.detect(frame_bgr, open_vocab_labels=active_labels)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[: self.max_detections]
        masks = self.segmenter.segment(frame_bgr, detections)
        embeddings = self.embedder.encode(frame_bgr, detections)
        n = min(len(detections), int(embeddings.shape[0]) if embeddings.ndim > 0 else 0)
        if n < len(detections):
            detections = detections[:n]
            masks = masks[:n]
            embeddings = embeddings[:n]
        for idx, det in enumerate(detections):
            det.embedding_idx = idx
        return PerceptionOutput(detections=detections, masks=masks, embeddings=embeddings)
