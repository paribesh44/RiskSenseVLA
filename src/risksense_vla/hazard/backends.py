"""Backend abstractions for prompt-driven hazard VLM reasoning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_LOGGER = logging.getLogger(__name__)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to the given bounds [low, high]."""
    return max(low, min(high, value))


def _normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a vector, with eps to avoid division by zero."""
    return vec / (torch.linalg.norm(vec) + eps)


def _text_proto(text: str, dim: int) -> torch.Tensor:
    """Produce a deterministic unit-norm embedding from text via SHA256-seeded randn."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    generator = torch.Generator()
    generator.manual_seed(seed)
    vec = torch.randn((dim,), generator=generator, dtype=torch.float32)
    return _normalize(vec)


def _fit_vec(vec: torch.Tensor | None, dim: int) -> torch.Tensor:
    """Pad or truncate a vector to the target dimension; returns zeros if vec is None or empty."""
    out = torch.zeros((dim,), dtype=torch.float32)
    if vec is None or vec.numel() == 0:
        return out
    flat = vec.detach().to(torch.float32).flatten()
    n = min(dim, flat.shape[0])
    out[:n] = flat[:n]
    return out


@dataclass(slots=True)
class VLMOutput:
    """Output from a VLM backend: generated text, inference timing, and optional metadata."""

    generated_text: str
    inference_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HazardConfig:
    """Configuration for hazard VLM backends: thresholds, model paths, and inference settings."""

    alert_threshold: float = 0.65
    backend_type: str = "phi4_mm"
    max_tokens: int = 64
    temperature: float = 0.2
    quantized: bool = True
    explain: bool = True
    debug_prompt: bool = False
    checkpoint_path: str | None = "artifacts/hazard_reasoner.pt"
    emb_dim: int = 256
    fallback_mode: str = "blend"
    lightweight_mode: bool = False
    phi4_model_id: str = "microsoft/Phi-4-multimodal-instruct"
    phi4_precision: str = "int8"
    phi4_estimated_vram_gb: float = 10.0


class BaseVLMBackend(ABC):
    """Abstract interface for hazard VLM backends."""

    @abstractmethod
    def predict_risk(
        self,
        prompt: str,
        image: np.ndarray | None,
        hoi_embedding: torch.Tensor | None = None,
        future_embedding: torch.Tensor | None = None,
        memory_embedding: torch.Tensor | None = None,
    ) -> VLMOutput:
        raise NotImplementedError

    def predict_risks(
        self,
        prompts: list[str],
        image: np.ndarray | None,
        hoi_embeddings: list[torch.Tensor] | None = None,
        future_embeddings: list[torch.Tensor] | None = None,
        memory_embedding: torch.Tensor | None = None,
    ) -> list[VLMOutput]:
        out: list[VLMOutput] = []
        for idx, prompt in enumerate(prompts):
            hoi_emb = hoi_embeddings[idx] if hoi_embeddings is not None and idx < len(hoi_embeddings) else None
            fut_emb = future_embeddings[idx] if future_embeddings is not None and idx < len(future_embeddings) else None
            out.append(
                self.predict_risk(
                    prompt=prompt,
                    image=image,
                    hoi_embedding=hoi_emb,
                    future_embedding=fut_emb,
                    memory_embedding=memory_embedding,
                )
            )
        return out

    def backend_metadata(self) -> dict[str, Any]:
        return {}


class _TinyHazardNet(nn.Module):
    """Small distilled net compatible with scripts/train_hazard_vlm.py outputs."""

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyLocalVLMBackend(BaseVLMBackend):
    """Lightweight local fallback backend used only in lightweight mode."""

    def __init__(self, config: HazardConfig):
        self.config = config
        self._model: nn.Module | None = None
        self._model_loaded = False

    def _load_model(self) -> None:
        if self._model_loaded:
            return
        self._model_loaded = True
        model = _TinyHazardNet(emb_dim=self.config.emb_dim)
        ckpt = self.config.checkpoint_path
        if ckpt:
            path = Path(ckpt)
            if path.exists():
                try:
                    payload = torch.load(path, map_location="cpu")
                    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else {}
                    if isinstance(state_dict, dict) and state_dict:
                        model.load_state_dict(state_dict, strict=False)
                except Exception as exc:  # pragma: no cover
                    _LOGGER.warning("hazard tiny backend failed to load checkpoint (%s); using initialized weights", exc)
        if self.config.quantized:
            try:
                model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            except Exception as exc:  # pragma: no cover
                _LOGGER.info("hazard tiny backend quantization unavailable: %s", exc)
        model.eval()
        self._model = model

    def _compose_feature(
        self,
        prompt: str,
        image: np.ndarray | None,
        hoi_embedding: torch.Tensor | None,
        future_embedding: torch.Tensor | None,
        memory_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        feature = _text_proto(prompt, self.config.emb_dim)
        if image is not None and image.size > 0:
            img = image.astype(np.float32)
            feature = feature.clone()
            feature[0] = _clamp(0.5 * (feature[0].item() + float(np.mean(img) / 255.0)))
            feature[1] = _clamp(0.5 * (feature[1].item() + float(np.std(img) / 128.0)))
        hoi_vec = _fit_vec(hoi_embedding, self.config.emb_dim)
        fut_vec = _fit_vec(future_embedding, self.config.emb_dim)
        mem_vec = _fit_vec(memory_embedding, self.config.emb_dim)
        fused = _normalize(0.35 * feature + 0.30 * hoi_vec + 0.20 * fut_vec + 0.15 * mem_vec)
        return fused.unsqueeze(0)

    def predict_risk(
        self,
        prompt: str,
        image: np.ndarray | None,
        hoi_embedding: torch.Tensor | None = None,
        future_embedding: torch.Tensor | None = None,
        memory_embedding: torch.Tensor | None = None,
    ) -> VLMOutput:
        self._load_model()
        start = time.perf_counter()
        feature = self._compose_feature(prompt, image, hoi_embedding, future_embedding, memory_embedding)
        score = 0.35
        if self._model is not None:
            with torch.no_grad():
                logits = self._model(feature)[0]
                probs = torch.softmax(logits, dim=0)
                score = float(0.15 * probs[0] + 0.55 * probs[1] + 0.92 * probs[2])
        score = _clamp(float(score))
        text = (
            f"Risk score: {score:.3f}\n"
            "Explanation: lightweight multimodal adapter estimates risk from prompt, "
            "current interaction embedding, predicted interaction embedding, and memory context."
        )
        return VLMOutput(
            generated_text=text,
            inference_ms=(time.perf_counter() - start) * 1000.0,
            metadata={"backend": "tiny", "lightweight_mode": True},
        )

    def backend_metadata(self) -> dict[str, Any]:
        return {"backend": "tiny", "mode": "lightweight"}


class Phi4MultimodalBackend(BaseVLMBackend):
    """Primary Phase-4 multimodal backend based on Phi-4 MM."""

    def __init__(self, config: HazardConfig):
        self.config = config
        self._pipeline: Any | None = None  # HF pipeline type varies by task
        self._init_attempted = False

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        if self._init_attempted:
            raise RuntimeError("Phi-4 multimodal backend initialization previously failed.")
        self._init_attempted = True
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="image-text-to-text",
                model=self.config.phi4_model_id,
                trust_remote_code=True,
                device_map="auto",
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to initialize Phi-4 multimodal backend ({self.config.phi4_model_id}): {exc}"
            ) from exc

    def _augment_prompt(
        self,
        prompt: str,
        hoi_embedding: torch.Tensor | None,
        future_embedding: torch.Tensor | None,
        memory_embedding: torch.Tensor | None,
    ) -> str:
        hoi_vec = _fit_vec(hoi_embedding, self.config.emb_dim)
        fut_vec = _fit_vec(future_embedding, self.config.emb_dim)
        mem_vec = _fit_vec(memory_embedding, self.config.emb_dim)
        cosine_hf = float(torch.dot(_normalize(hoi_vec), _normalize(fut_vec)).item()) if fut_vec.numel() else 0.0
        cosine_hm = float(torch.dot(_normalize(hoi_vec), _normalize(mem_vec)).item()) if mem_vec.numel() else 0.0
        return (
            f"{prompt}\n\n"
            "Embedding context:\n"
            f"- hoi_embedding_norm: {float(torch.linalg.norm(hoi_vec).item()):.4f}\n"
            f"- future_embedding_norm: {float(torch.linalg.norm(fut_vec).item()):.4f}\n"
            f"- memory_embedding_norm: {float(torch.linalg.norm(mem_vec).item()):.4f}\n"
            f"- hoi_future_cosine: {cosine_hf:.4f}\n"
            f"- hoi_memory_cosine: {cosine_hm:.4f}\n\n"
            "Return exactly:\n"
            "Risk score: <float 0-1>\n"
            "Explanation: <short text>"
        )

    def _extract_generated_text(self, raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            for key in ("generated_text", "text", "answer"):
                if key in raw and isinstance(raw[key], str):
                    return raw[key]
            return str(raw)
        if isinstance(raw, list) and raw:
            return self._extract_generated_text(raw[0])
        return str(raw)

    def predict_risk(
        self,
        prompt: str,
        image: np.ndarray | None,
        hoi_embedding: torch.Tensor | None = None,
        future_embedding: torch.Tensor | None = None,
        memory_embedding: torch.Tensor | None = None,
    ) -> VLMOutput:
        self._ensure_pipeline()
        assert self._pipeline is not None
        start = time.perf_counter()
        prompt_text = self._augment_prompt(prompt, hoi_embedding, future_embedding, memory_embedding)
        image_rgb = image[..., ::-1] if image is not None else None
        try:
            if image_rgb is not None:
                raw = self._pipeline(
                    text=prompt_text,
                    images=image_rgb,
                    max_new_tokens=int(self.config.max_tokens),
                    temperature=float(self.config.temperature),
                )
            else:
                raw = self._pipeline(
                    text=prompt_text,
                    max_new_tokens=int(self.config.max_tokens),
                    temperature=float(self.config.temperature),
                )
        except TypeError:
            raw = self._pipeline(prompt_text)
        generated = self._extract_generated_text(raw).strip()
        if "Risk score" not in generated:
            generated = f"Risk score: 0.50\nExplanation: {generated[:220]}"
        return VLMOutput(
            generated_text=generated,
            inference_ms=(time.perf_counter() - start) * 1000.0,
            metadata={
                "backend": "phi4_mm",
                "model_id": self.config.phi4_model_id,
                "precision": self.config.phi4_precision,
                "estimated_vram_gb": self.config.phi4_estimated_vram_gb,
            },
        )

    def backend_metadata(self) -> dict[str, Any]:
        return {
            "backend": "phi4_mm",
            "model_id": self.config.phi4_model_id,
            "precision": self.config.phi4_precision,
            "estimated_vram_gb": self.config.phi4_estimated_vram_gb,
        }


class StubBackend(BaseVLMBackend):
    """Deterministic backend for CI and explicit lightweight fallback."""

    def __init__(self, config: HazardConfig):
        self.config = config

    def _deterministic_score(self, prompt: str) -> float:
        text = prompt.lower()
        base = 0.2
        if "touch_hot_surface" in text or "hot" in text:
            base += 0.52
        if "cut" in text or "knife" in text:
            base += 0.45
        if "vehicle" in text:
            base += 0.48
        if "predicted interaction: yes" in text:
            base += 0.08
        confidence_match = re.search(r"interaction_confidence:\s*(\d*\.?\d+)", text)
        if confidence_match:
            conf = _clamp(float(confidence_match.group(1)))
            base = 0.55 * base + 0.45 * conf
        return _clamp(base)

    def predict_risk(
        self,
        prompt: str,
        image: np.ndarray | None,
        hoi_embedding: torch.Tensor | None = None,
        future_embedding: torch.Tensor | None = None,
        memory_embedding: torch.Tensor | None = None,
    ) -> VLMOutput:
        del image, hoi_embedding, future_embedding, memory_embedding
        start = time.perf_counter()
        score = self._deterministic_score(prompt)
        explanation = (
            f"deterministic stub analysis estimates {score:.2f} risk from action/object tokens "
            "and confidence in memory context."
        )
        text = f"Risk score: {score:.3f}\nExplanation: {explanation}"
        return VLMOutput(
            generated_text=text,
            inference_ms=(time.perf_counter() - start) * 1000.0,
            metadata={"backend": "stub", "source": "deterministic"},
        )

    def backend_metadata(self) -> dict[str, Any]:
        return {"backend": "stub", "mode": "deterministic"}
