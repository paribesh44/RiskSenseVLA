"""Frame rendering backends for synthetic hazard scenes."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from risksense_vla.synthetic.scene_config import ROOM_PRESETS, SceneConfig
from risksense_vla.synthetic.sequence_engine import AnnotatedFrame

logger = logging.getLogger(__name__)

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (50, 120, 200),
    "toddler": (80, 180, 255),
    "knife": (0, 0, 200),
    "stove": (0, 80, 200),
    "hot_pan": (0, 100, 220),
    "glass": (200, 200, 100),
    "bottle": (180, 160, 60),
    "chemical_bottle": (0, 180, 0),
    "electrical_cord": (100, 100, 100),
    "wet_floor": (200, 180, 50),
    "sharp_scissors": (60, 60, 180),
    "vehicle": (120, 80, 40),
}


@runtime_checkable
class RendererProtocol(Protocol):
    def render_frame(
        self, scene_config: SceneConfig, frame: AnnotatedFrame
    ) -> np.ndarray: ...


class ProceduralRenderer:
    """Renders simple geometric scenes using OpenCV primitives."""

    def render_frame(
        self, scene_config: SceneConfig, frame: AnnotatedFrame
    ) -> np.ndarray:
        w, h = scene_config.resolution
        preset = ROOM_PRESETS.get(scene_config.room_type, {})
        bg = preset.get("bg_color", (200, 200, 200))
        canvas = np.full((h, w, 3), bg, dtype=np.uint8)

        for obj in frame.objects:
            bbox = obj.get("bbox_xyxy", [0, 0, 40, 40])
            label = obj.get("label", "object")
            color = _LABEL_COLORS.get(label, (150, 150, 150))
            x1, y1, x2, y2 = (
                max(0, min(bbox[0], w - 1)),
                max(0, min(bbox[1], h - 1)),
                max(0, min(bbox[2], w - 1)),
                max(0, min(bbox[3], h - 1)),
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(
                canvas,
                label[:10],
                (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if frame.occluded:
            overlay = np.zeros_like(canvas, dtype=np.uint8)
            overlay[:] = (60, 60, 60)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        return canvas


class StableDiffusionRenderer:
    """Generates frames via Stable Diffusion (requires ``diffusers`` extra)."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        num_inference_steps: int = 25,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.num_inference_steps = num_inference_steps
        self._pipe: object | None = None

    def _ensure_pipeline(self) -> object:
        if self._pipe is not None:
            return self._pipe
        try:
            from diffusers import StableDiffusionPipeline  # type: ignore[import-untyped]

            import torch

            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            )
            pipe = pipe.to(self.device)
            self._pipe = pipe
            return pipe
        except ImportError:
            raise ImportError(
                "diffusers is required for StableDiffusionRenderer. "
                "Install with: pip install risksense-vla[synthetic]"
            )

    def render_frame(
        self, scene_config: SceneConfig, frame: AnnotatedFrame
    ) -> np.ndarray:
        pipe = self._ensure_pipeline()
        prompt = self._build_prompt(scene_config, frame)
        w, h = scene_config.resolution
        w = (w // 8) * 8
        h = (h // 8) * 8

        result = pipe(  # type: ignore[operator]
            prompt,
            width=w,
            height=h,
            num_inference_steps=self.num_inference_steps,
        )
        pil_image = result.images[0]  # type: ignore[attr-defined]
        rgb = np.array(pil_image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _build_prompt(cfg: SceneConfig, frame: AnnotatedFrame) -> str:
        obj_labels = [o["label"] for o in frame.objects]
        hoi = frame.hoi
        parts = [
            f"A {cfg.room_type} scene",
            f"with {', '.join(obj_labels)}",
            f"where a {hoi.get('subject', 'person')} is {hoi.get('action', 'interacting')}",
            f"with a {hoi.get('object', 'object')}",
            f"{cfg.lighting} lighting",
            "photorealistic, indoor, high detail",
        ]
        return ", ".join(parts)


def get_renderer(name: str, **kwargs: object) -> RendererProtocol:
    """Factory for renderer selection."""
    if name == "sd":
        return StableDiffusionRenderer(**kwargs)  # type: ignore[arg-type]
    return ProceduralRenderer()
