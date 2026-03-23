"""Phase 1 internal architecture for open-vocabulary perception.

SECTION 1 - Architectural Role
This module implements open-vocabulary detection and box-conditioned segmentation for
the Hazard-Aware Predictive VLA perception stage, then produces CLIP-aligned object
embeddings for downstream reasoning. Its outputs are consumed by hazard-aware memory
and predictive HOI modules as frame-level perception primitives; temporal aggregation,
memory update policy, and HOI forecasting logic are explicitly out of scope here.

SECTION 2 - Detection Pipeline
Per-frame processing follows:
frame -> open-vocabulary detector -> bounding boxes -> segmentation masks ->
region crops -> CLIP encoder -> embedding vectors.
For detection i:
e_i = CLIP(f(mask_i \\odot frame))
where mask_i isolates the object region, \\odot is element-wise masking, f(.) denotes
deterministic preprocessing (crop/resize/normalize/color conversion), and CLIP(.) is a
frozen image encoder (or deterministic fallback encoder if CLIP runtime is unavailable).

SECTION 3 - Open-Vocabulary Handling
Dynamic label prompts are passed through infer(..., labels=...) and forwarded to
detector backends as open_vocab_labels, enabling zero-shot detection against runtime
text classes rather than a fixed closed-set taxonomy. If no labels are provided, or if
provided labels collapse to empty tokens after sanitization, backend default labels are
used to preserve deterministic API behavior.

SECTION 4 - Output Contract
The module returns a canonical list[PerceptionDetection]. Each item includes:
- bounding_box: bbox_xyxy as tuple[int, int, int, int] in pixel coordinates.
- mask: per-object spatial region estimate for the same object.
- clip_embedding: fixed-width float32 vector.
- confidence: float in [0, 1].
Dimension guarantee: clip_embedding width is d = 256 in the default Phase 1
configuration (and remains constant within a run if reconfigured).
Additional guarantees: embeddings are L2-normalized and each detection carries its own
mask + embedding (no downstream tensor indexing contract).

SECTION 5 - Performance Assumptions
Designed for >=20 FPS on a laptop-class GPU under typical scene complexity.
A CPU fallback path exists for detector/embedder backends but may reduce throughput.
No temporal state is retained for scene dynamics in this module; each infer call is
frame-local computation.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch

from risksense_vla.perception.embed import CLIPEmbedder, FallbackEmbedder
from risksense_vla.perception.segment import BoxMaskSegmenter
from risksense_vla.types import Detection, PerceptionDetection

_LOGGER = logging.getLogger(__name__)


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


def _looks_like_hf_repo_id(model_name: str) -> bool:
    """True when ``model_name`` is likely a Hugging Face repo id, not a local path."""
    if not model_name or "/" not in model_name:
        return False
    if os.path.isabs(model_name) and os.path.isdir(model_name):
        return False
    if os.path.isdir(model_name):
        return False
    return True


def _remove_zero_byte_incomplete_blobs(repo_id: str) -> int:
    """Clean stale 0-byte downloads that can leave hub lock state wedged."""
    hub_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )
    blob_dir = Path(hub_root) / ("models--" + repo_id.replace("/", "--")) / "blobs"
    if not blob_dir.is_dir():
        return 0
    removed = 0
    for incomplete_blob in blob_dir.glob("*.incomplete"):
        try:
            if incomplete_blob.stat().st_size == 0:
                incomplete_blob.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _download_grounding_dino_weight_files(repo_id: str) -> None:
    """Pre-download checkpoints so model load does not appear hung on first frame."""
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    removed = _remove_zero_byte_incomplete_blobs(repo_id)
    if removed:
        _LOGGER.warning(
            "Removed %s stale 0-byte incomplete file(s) from Hugging Face cache.",
            removed,
        )

    last_err: Exception | None = None
    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            _LOGGER.info(
                "Downloading / verifying Grounding DINO weights: %s (%s) ...",
                repo_id,
                filename,
            )
            hf_hub_download(repo_id=repo_id, filename=filename)
            return
        except Exception as exc:  # pragma: no cover - network/hub environment dependent
            last_err = exc
    if last_err is not None:
        raise RuntimeError(
            "Unable to download Grounding DINO checkpoint from Hugging Face Hub. "
            "Set HF_TOKEN, remove stale '*.incomplete' files from the model cache, "
            "or run with the YOLOE backend config to avoid this dependency."
        ) from last_err


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
    _model: Any = None  # HF AutoModelForZeroShotObjectDetection type
    _processor: Any = None  # HF AutoProcessor type
    _init_failed: bool = False
    _init_error: str = ""

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if self._init_failed:
            raise RuntimeError(self._init_error or "Grounding DINO initialization previously failed.")
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            if not self.local_files_only and _looks_like_hf_repo_id(self.model_name):
                _download_grounding_dino_weight_files(self.model_name)

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
        # GroundingDINO in some Transformers versions expects a single caption-like
        # prompt per image (not a list of independent class strings).
        text_prompt = " . ".join(labels).strip()
        if text_prompt and not text_prompt.endswith("."):
            text_prompt += "."
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=frame_rgb, text=text_prompt, return_tensors="pt")
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
    _model: Any = None  # Ultralytics YOLO type
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
class OpenVocabPerception:
    detector: Detector
    segmenter: BoxMaskSegmenter
    embedder: Embedder
    default_labels: list[str] = field(default_factory=lambda: ["person", "knife", "stove", "vehicle"])
    max_detections: int = 12
    allow_mock_backend: bool = False
    fallback_detector: Detector = field(default_factory=MockOpenVocabDetector)

    @classmethod
    def default(cls) -> "OpenVocabPerception":
        return cls.from_config(cfg={})

    @classmethod
    def from_config(cls, cfg: dict[str, Any], device: str = "cpu") -> "OpenVocabPerception":
        p_cfg = cfg.get("perception", {}) if isinstance(cfg, dict) else {}
        m_cfg = cfg.get("models", {}) if isinstance(cfg, dict) else {}
        detector_name = str(p_cfg.get("detector_backend", m_cfg.get("detector", "grounding_dino"))).lower()
        embedder_name = str(p_cfg.get("embedder_backend", m_cfg.get("embedder", "clip_or_fallback"))).lower()
        det_dev_mode = str(p_cfg.get("detector_device", "auto")).lower()
        emb_dev_mode = str(p_cfg.get("embedder_device", "auto")).lower()
        allow_mock_backend = bool(p_cfg.get("allow_mock_backend", False))
        max_detections = int(p_cfg.get("detector_max_detections", 12))
        labels = p_cfg.get("default_labels", ["person", "knife", "stove", "vehicle"])
        labels = [str(x) for x in labels]
        if detector_name == "mock" and not allow_mock_backend:
            raise ValueError(
                "Mock detector requested but disabled by config. "
                "Set perception.allow_mock_backend=true to enable it explicitly."
            )
        if det_dev_mode == "auto":
            if detector_name in {"grounding_dino", "groundingdino"} and str(device) == "mps":
                detector_device = "cpu"
                _LOGGER.info(
                    "Grounding DINO runs on CPU (MPS support for this checkpoint is unreliable)."
                )
            else:
                detector_device = str(device)
        else:
            detector_device = det_dev_mode
        embedder_device = str(device) if emb_dev_mode == "auto" else emb_dev_mode

        if detector_name in {"grounding_dino", "groundingdino"}:
            detector: Detector = GroundingDINOAdapter(
                model_name=str(p_cfg.get("grounding_dino_model_id", "IDEA-Research/grounding-dino-base")),
                box_threshold=float(p_cfg.get("detector_confidence_threshold", 0.35)),
                text_threshold=float(p_cfg.get("detector_text_threshold", 0.25)),
                max_detections=max_detections,
                default_labels=labels,
                device=detector_device,
                local_files_only=bool(p_cfg.get("local_files_only", False)),
            )
        elif detector_name in {"yoloe26", "yoloe-26", "yoloe"}:
            detector = YOLOE26Adapter(
                model_path=str(p_cfg.get("yoloe_model_path", "yoloe-26s.pt")),
                confidence_threshold=float(p_cfg.get("detector_confidence_threshold", 0.35)),
                max_detections=max_detections,
                default_labels=labels,
                device=detector_device,
            )
        else:
            detector = MockOpenVocabDetector(default_label=labels[0] if labels else "person")

        embedding_dim = int(p_cfg.get("embedding_dim", 256))
        if embedder_name == "clip":
            embedder: Embedder = CLIPEmbedder(
                model_name=str(p_cfg.get("clip_model_id", "openai/clip-vit-base-patch32")),
                output_dim=embedding_dim,
                device=embedder_device,
                batch_size=int(p_cfg.get("clip_batch_size", 8)),
                enabled=True,
                allow_fallback=False,
                local_files_only=bool(p_cfg.get("local_files_only", False)),
            )
        elif embedder_name in {"clip_or_fallback", "clip-fallback"}:
            embedder = CLIPEmbedder(
                model_name=str(p_cfg.get("clip_model_id", "openai/clip-vit-base-patch32")),
                output_dim=embedding_dim,
                device=embedder_device,
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
            allow_mock_backend=allow_mock_backend,
            fallback_detector=MockOpenVocabDetector(default_label=labels[0] if labels else "person"),
        )

    def infer(self, frame_bgr: np.ndarray, labels: list[str] | None = None) -> list[PerceptionDetection]:
        active_labels = labels if labels is not None else self.default_labels
        try:
            detections = self.detector.detect(frame_bgr, open_vocab_labels=active_labels)
        except Exception as exc:
            if not self.allow_mock_backend:
                raise RuntimeError(
                    "Detector inference failed and mock fallback is disabled. "
                    "Set perception.allow_mock_backend=true to allow explicit mock fallback."
                ) from exc
            detections = self.fallback_detector.detect(frame_bgr, open_vocab_labels=active_labels)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[: self.max_detections]
        masks = self.segmenter.segment(frame_bgr, detections)
        embeddings = self.embedder.encode(frame_bgr, detections)
        n = min(len(detections), int(embeddings.shape[0]) if embeddings.ndim > 0 else 0)
        if n < len(detections):
            detections = detections[:n]
            masks = masks[:n]
            embeddings = embeddings[:n]
        out: list[PerceptionDetection] = []
        for idx, det in enumerate(detections):
            out.append(
                PerceptionDetection(
                    track_id=det.track_id,
                    label=det.label,
                    confidence=float(det.confidence),
                    bbox_xyxy=tuple(det.bbox_xyxy),
                    mask=masks[idx].detach().clone(),
                    clip_embedding=embeddings[idx].detach().clone(),
                )
            )
        return out
