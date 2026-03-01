"""Segmentation adapters for detected objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hapvla.types import Detection


@dataclass(slots=True)
class BoxMaskSegmenter:
    """Fast mask generation from detection boxes."""

    mask_dtype: torch.dtype = torch.float32

    def segment(self, frame_bgr: np.ndarray, detections: list[Detection]) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        if not detections:
            return torch.zeros((0, h, w), dtype=self.mask_dtype)
        masks = torch.zeros((len(detections), h, w), dtype=self.mask_dtype)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox_xyxy
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                masks[i, y1:y2, x1:x2] = 1.0
        return masks
