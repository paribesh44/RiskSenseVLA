"""Object embedding extraction with CLIP and lightweight fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from risksense_vla.types import Detection


def _crop(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return frame_bgr[y1:y2, x1:x2]


def _fit_embedding_dim(embeddings: torch.Tensor, output_dim: int) -> torch.Tensor:
    if embeddings.numel() == 0:
        return torch.zeros((0, output_dim), dtype=torch.float32)
    out = embeddings.to(dtype=torch.float32)
    if out.shape[1] > output_dim:
        out = out[:, :output_dim]
    elif out.shape[1] < output_dim:
        out = F.pad(out, (0, output_dim - out.shape[1]))
    norm = torch.linalg.norm(out, dim=1, keepdim=True) + 1e-8
    return out / norm


@dataclass(slots=True)
class FallbackEmbedder:
    """Histogram-based embedding for deterministic low-latency fallback."""

    dim: int = 256

    def encode(self, frame_bgr: np.ndarray, detections: list[Detection]) -> torch.Tensor:
        if not detections:
            return torch.zeros((0, self.dim), dtype=torch.float32)
        embs = []
        for det in detections:
            crop = _crop(frame_bgr, det.bbox_xyxy)
            crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
            hist_b = cv2.calcHist([crop], [0], None, [16], [0, 256]).flatten()
            hist_g = cv2.calcHist([crop], [1], None, [16], [0, 256]).flatten()
            hist_r = cv2.calcHist([crop], [2], None, [16], [0, 256]).flatten()
            vec = np.concatenate([hist_b, hist_g, hist_r], axis=0).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            tiled = np.tile(vec, int(np.ceil(self.dim / len(vec))))[: self.dim]
            embs.append(tiled)
        return torch.tensor(np.stack(embs), dtype=torch.float32)


@dataclass(slots=True)
class CLIPEmbedder:
    """CLIP image embedding backend with graceful fallback."""

    model_name: str = "openai/clip-vit-base-patch32"
    output_dim: int = 256
    device: str = "cpu"
    batch_size: int = 8
    crop_size: int = 224
    enabled: bool = True
    allow_fallback: bool = True
    local_files_only: bool = False
    _model: Any = None  # HF CLIPModel type
    _processor: Any = None  # HF CLIPProcessor type
    _fallback: FallbackEmbedder | None = None
    _init_failed: bool = False
    _init_error: str = ""

    def _ensure_model(self) -> None:
        if not self.enabled:
            return
        if self._model is not None and self._processor is not None:
            return
        if self._init_failed:
            raise RuntimeError(self._init_error or "CLIP initialization previously failed.")
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            )
            self._model = CLIPModel.from_pretrained(self.model_name, local_files_only=self.local_files_only)
            self._model.eval()
            self._model.to(self.device)
        except Exception as exc:
            self._init_failed = True
            self._init_error = str(exc)
            raise

    def _fallback_encode(self, frame_bgr: np.ndarray, detections: list[Detection]) -> torch.Tensor:
        if self._fallback is None:
            self._fallback = FallbackEmbedder(dim=self.output_dim)
        return self._fallback.encode(frame_bgr, detections)

    def encode(self, frame_bgr: np.ndarray, detections: list[Detection]) -> torch.Tensor:
        if not detections:
            return torch.zeros((0, self.output_dim), dtype=torch.float32)

        if not self.enabled:
            return self._fallback_encode(frame_bgr, detections)

        try:
            self._ensure_model()
            assert self._model is not None and self._processor is not None
            crops_rgb: list[np.ndarray] = []
            for det in detections:
                crop = _crop(frame_bgr, det.bbox_xyxy)
                crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops_rgb.append(crop_rgb)

            feats: list[torch.Tensor] = []
            step = max(1, self.batch_size)
            for i in range(0, len(crops_rgb), step):
                batch = crops_rgb[i : i + step]
                inputs = self._processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                with torch.inference_mode():
                    batch_feats = self._model.get_image_features(pixel_values=pixel_values)
                batch_feats = batch_feats / (torch.linalg.norm(batch_feats, dim=1, keepdim=True) + 1e-8)
                feats.append(batch_feats.detach().cpu())

            emb = torch.cat(feats, dim=0)
            return _fit_embedding_dim(emb, self.output_dim)
        except Exception:
            if self.allow_fallback:
                return self._fallback_encode(frame_bgr, detections)
            raise
