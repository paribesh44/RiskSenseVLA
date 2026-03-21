"""Prompt-driven hazard reasoner with pluggable VLM backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import math
import re
import time
from typing import Iterable

import numpy as np
import torch

from risksense_vla.types import HOITriplet, HazardScore, MemoryObjectState, MemoryState

from .backends import (
    BaseVLMBackend,
    HazardConfig,
    Phi4MultimodalBackend,
    SmolVlmBackend,
    StubBackend,
    TinyLocalVLMBackend,
    VLMOutput,
)

_LOGGER = logging.getLogger(__name__)

HOIKey = str


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to the given bounds [low, high]."""
    return max(low, min(high, value))


@dataclass(slots=True)
class HazardOutput:
    """Result of hazard reasoning: per-HOI scores, alerts, explanations, and global risk."""

    hazards: list[HazardScore]
    alerts: list[str]
    hazard_by_track_id: dict[str, float]
    hazard_by_hoi: dict[HOIKey, float]
    explanations: dict[str, str]
    global_risk_score: float
    inference_ms: float = 0.0
    prompt_debug: dict[HOIKey, str] = field(default_factory=dict)
    backend: str = "unknown"
    backend_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def hazard_map(self) -> dict[str, float]:
        return self.hazard_by_track_id

    @property
    def hazard_map_legacy(self) -> dict[HOIKey, float]:
        return self.hazard_by_hoi

    @property
    def hazard_alerts(self) -> list[str]:
        return self.alerts


# Backward compatible name preserved for external imports.
HazardReasoningOutput = HazardOutput


class HazardReasoner:
    """Canonical prompt-driven hazard reasoner with swappable VLM backend."""

    def __init__(self, backend: BaseVLMBackend, config: HazardConfig):
        self.backend = backend
        self.config = config

    def _severity(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    def _coerce_triplet(self, hoi: object) -> HOITriplet:
        if isinstance(hoi, HOITriplet):
            return hoi
        required = ("subject", "action", "object", "confidence", "t_start", "t_end")
        for field_name in required:
            if not hasattr(hoi, field_name):
                raise TypeError(f"Unsupported HOI item: missing attribute '{field_name}'")
        return HOITriplet(
            subject=str(getattr(hoi, "subject")),
            action=str(getattr(hoi, "action")),
            object=str(getattr(hoi, "object")),
            confidence=float(getattr(hoi, "confidence")),
            t_start=float(getattr(hoi, "t_start")),
            t_end=float(getattr(hoi, "t_end")),
            predicted=bool(getattr(hoi, "predicted", False)),
        )

    def _resolve_track_object(self, object_label: str, memory_state: MemoryState) -> MemoryObjectState | None:
        label = object_label.strip().lower()
        matches = [obj for obj in memory_state.objects if obj.label.strip().lower() == label]
        if not matches:
            return None
        return max(matches, key=lambda x: (x.persistence + 0.25 * x.hazard_weight, x.age_frames))

    def _resolve_track_id(self, object_label: str, memory_state: MemoryState) -> str:
        target = self._resolve_track_object(object_label=object_label, memory_state=memory_state)
        if target is None:
            return f"unknown:{object_label.strip().lower()}"
        return target.track_id

    def _bbox_center(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _bbox_iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _proximity_flags(self, target: MemoryObjectState | None, memory_state: MemoryState) -> list[str]:
        if target is None:
            return ["track_unresolved"]
        flags: list[str] = []
        tx, ty = self._bbox_center(target.last_bbox_xyxy)
        for obj in memory_state.objects:
            if obj.track_id == target.track_id:
                continue
            iou = self._bbox_iou(target.last_bbox_xyxy, obj.last_bbox_xyxy)
            ox, oy = self._bbox_center(obj.last_bbox_xyxy)
            dist = math.sqrt((tx - ox) ** 2 + (ty - oy) ** 2)
            if iou >= 0.10:
                flags.append(f"overlap_with_{obj.label}")
            elif dist <= 90.0:
                flags.append(f"near_{obj.label}")
            if obj.hazard_weight >= 0.65 and (iou > 0.0 or dist <= 110.0):
                flags.append("close_to_hazardous_object")
        return sorted(set(flags)) if flags else ["no_close_neighbors"]

    def _summarize_memory(self, memory_state: MemoryState, target: MemoryObjectState | None) -> str:
        if not memory_state.objects:
            return "No persistent objects tracked. hazard_history: unavailable. velocity: unavailable."
        top = sorted(
            memory_state.objects,
            key=lambda x: (x.hazard_weight, x.persistence, x.age_frames),
            reverse=True,
        )[:5]
        chunks = []
        for obj in top:
            chunks.append(
                f"{obj.track_id}:{obj.label}(persistence={obj.persistence:.2f},"
                f"hazard_history={obj.hazard_weight:.2f},age_frames={obj.age_frames})"
            )
        velocity = "unavailable"
        if target is not None:
            velocity = f"unavailable_for_{target.track_id}_single_state_only"
        return "Persistent objects: " + "; ".join(chunks) + f". motion_velocity: {velocity}."

    def _build_prompt(
        self,
        hoi: HOITriplet,
        track_id: str,
        memory_summary: str,
        proximity_flags: list[str],
        future_embedding_norm: float,
    ) -> str:
        pred = "yes" if hoi.predicted else "no"
        prox_text = ", ".join(proximity_flags)
        return (
            "You are a safety reasoning system.\n\n"
            "Scene context:\n"
            f"- {memory_summary}\n"
            f"- Current interaction: {hoi.subject} is {hoi.action} {hoi.object}\n"
            f"- Track id: {track_id}\n"
            f"- Predicted interaction: {pred}\n"
            f"- Interaction_confidence: {_clamp(hoi.confidence):.3f}\n"
            f"- Object proximity risk factors: {prox_text}\n\n"
            f"- Future_interaction_embedding_norm: {future_embedding_norm:.4f}\n\n"
            "Question:\n"
            "On a scale from 0 to 1, how dangerous is this situation?\n"
            "Provide:\n"
            "1. Risk score (float 0-1)\n"
            "2. Short explanation.\n"
            "Format:\n"
            "Risk score: <float>\n"
            "Explanation: <text>"
        )

    def _parse_vlm_output(self, output: VLMOutput) -> tuple[float, str]:
        text = output.generated_text.strip()
        match = re.search(r"risk\s*score\s*:\s*(\d*\.?\d+)", text, flags=re.IGNORECASE)
        if match:
            score = _clamp(float(match.group(1)))
        else:
            float_match = re.search(r"(\d*\.?\d+)", text)
            score = _clamp(float(float_match.group(1))) if float_match else 0.35
        em = re.search(r"explanation\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if em:
            explanation = em.group(1).strip()
        else:
            explanation = "No explanation returned by backend; fallback explanation applied."
        if self.config.explain:
            explanation = explanation[:240]
        else:
            explanation = "Explanation disabled by config."
        return score, explanation

    def _text_proto(self, text: str) -> torch.Tensor:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        g = torch.Generator()
        g.manual_seed(seed)
        vec = torch.randn((self.config.emb_dim,), generator=g, dtype=torch.float32)
        return vec / (torch.linalg.norm(vec) + 1e-8)

    def _memory_embedding(self, memory_state: MemoryState) -> torch.Tensor:
        out = torch.zeros((self.config.emb_dim,), dtype=torch.float32)
        if memory_state.hoi_embedding.numel() == 0:
            return out
        flat = memory_state.hoi_embedding.detach().to(torch.float32).flatten()
        copy_n = min(self.config.emb_dim, flat.shape[0])
        out[:copy_n] = flat[:copy_n]
        return out

    def _future_embedding_for_index(self, idx: int, hoi_future_embeddings: torch.Tensor | None) -> torch.Tensor:
        if hoi_future_embeddings is None or hoi_future_embeddings.numel() == 0:
            return torch.zeros((self.config.emb_dim,), dtype=torch.float32)
        if hoi_future_embeddings.ndim != 3:
            return torch.zeros((self.config.emb_dim,), dtype=torch.float32)
        if idx >= hoi_future_embeddings.shape[0]:
            return torch.zeros((self.config.emb_dim,), dtype=torch.float32)
        row = hoi_future_embeddings[idx].detach().to(torch.float32)
        mean_row = row.mean(dim=0)
        out = torch.zeros((self.config.emb_dim,), dtype=torch.float32)
        copy_n = min(self.config.emb_dim, mean_row.shape[0])
        out[:copy_n] = mean_row[:copy_n]
        return out

    def predict_hazard(
        self,
        hoi_current: Iterable[object],
        hoi_future_embeddings: torch.Tensor | None,
        memory_state: MemoryState,
        frame_bgr: np.ndarray | None = None,
    ) -> HazardOutput:
        t0 = time.perf_counter()
        hois = [self._coerce_triplet(item) for item in hoi_current]
        if not hois:
            return HazardOutput(
                hazards=[],
                alerts=[],
                hazard_by_track_id={},
                hazard_by_hoi={},
                explanations={},
                global_risk_score=0.0,
                inference_ms=0.0,
                prompt_debug={},
                backend=self.config.backend_type,
                backend_metadata=self.backend.backend_metadata(),
            )

        prompts: list[str] = []
        track_ids: list[str] = []
        hoi_keys: list[str] = []
        hoi_embeddings: list[torch.Tensor] = []
        future_embeddings: list[torch.Tensor] = []
        for hoi in hois:
            idx = len(prompts)
            track = self._resolve_track_id(hoi.object, memory_state=memory_state)
            target = self._resolve_track_object(hoi.object, memory_state=memory_state)
            memory_summary = self._summarize_memory(memory_state=memory_state, target=target)
            prox = self._proximity_flags(target=target, memory_state=memory_state)
            hoi_emb = self._text_proto(f"{hoi.subject}:{hoi.action}:{hoi.object}")
            future_emb = self._future_embedding_for_index(idx, hoi_future_embeddings)
            future_norm = float(torch.linalg.norm(future_emb).item())
            prompts.append(
                self._build_prompt(
                    hoi=hoi,
                    track_id=track,
                    memory_summary=memory_summary,
                    proximity_flags=prox,
                    future_embedding_norm=future_norm,
                )
            )
            track_ids.append(track)
            hoi_keys.append(f"{hoi.subject}:{hoi.action}:{hoi.object}")
            hoi_embeddings.append(hoi_emb)
            future_embeddings.append(future_emb)

        outputs = self.backend.predict_risks(
            prompts=prompts,
            image=frame_bgr,
            hoi_embeddings=hoi_embeddings,
            future_embeddings=future_embeddings,
            memory_embedding=self._memory_embedding(memory_state),
        )
        hazards: list[HazardScore] = []
        alerts: list[str] = []
        hazard_by_track_id: dict[str, float] = {}
        hazard_by_hoi: dict[str, float] = {}
        explanations: dict[str, str] = {}
        prompt_debug: dict[str, str] = {}
        for idx, hoi in enumerate(hois):
            vlm_out = outputs[idx] if idx < len(outputs) else VLMOutput(generated_text="Risk score: 0.35\nExplanation: backend_missing")
            score, explanation = self._parse_vlm_output(vlm_out)
            severity = self._severity(score)
            track_id = track_ids[idx]
            hoi_key = hoi_keys[idx]
            hazards.append(
                HazardScore(
                    subject=hoi.subject,
                    action=hoi.action,
                    object=hoi.object,
                    score=score,
                    severity=severity,
                    explanation=explanation,
                )
            )
            hazard_by_track_id[track_id] = max(hazard_by_track_id.get(track_id, 0.0), score)
            hazard_by_hoi[hoi_key] = score
            if track_id not in explanations or score >= hazard_by_track_id.get(track_id, 0.0):
                explanations[track_id] = explanation
            if self.config.debug_prompt:
                prompt_debug[hoi_key] = prompts[idx]
            if score >= self.config.alert_threshold:
                horizon = "future" if hoi.predicted else "current"
                alerts.append(
                    f"[{severity.upper()}] {horizon} risk: {hoi.subject} may {hoi.action} {hoi.object} "
                    f"(track_id={track_id}, score={score:.2f})"
                )

        global_risk = _clamp(
            float(sum(hazard_by_track_id.values()) / len(hazard_by_track_id)) if hazard_by_track_id else 0.0
        )
        backend_meta: dict[str, object] = self.backend.backend_metadata()
        if outputs and outputs[0].metadata:
            backend_meta = {**backend_meta, **outputs[0].metadata}
        return HazardOutput(
            hazards=hazards,
            alerts=alerts,
            hazard_by_track_id=hazard_by_track_id,
            hazard_by_hoi=hazard_by_hoi,
            explanations=explanations,
            global_risk_score=global_risk,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
            prompt_debug=prompt_debug,
            backend=self.config.backend_type,
            backend_metadata=backend_meta,
        )

    def infer(self, hois: list[HOITriplet]) -> HazardOutput:
        timestamp = float(hois[0].t_start) if hois else 0.0
        return self.predict_hazard(
            hoi_current=hois,
            hoi_future_embeddings=None,
            memory_state=MemoryState(timestamp=timestamp),
        )


class DistilledHazardReasoner(HazardReasoner):
    """Backward-compatible wrapper with tiny/stub backend selection."""

    def __init__(
        self,
        alert_threshold: float = 0.65,
        checkpoint_path: str | None = "artifacts/hazard_reasoner.pt",
        fallback_mode: str = "blend",
        emb_dim: int = 256,
        backend_type: str = "phi4_mm",
        max_tokens: int = 64,
        temperature: float = 0.2,
        quantized: bool = True,
        lightweight_mode: bool = False,
        phi4_model_id: str = "microsoft/Phi-4-multimodal-instruct",
        phi4_precision: str = "int8",
        phi4_estimated_vram_gb: float = 10.0,
        vlm_model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        explain: bool = True,
        debug_prompt: bool = False,
        backend: BaseVLMBackend | None = None,
    ):
        cfg = HazardConfig(
            alert_threshold=float(alert_threshold),
            backend_type=str(backend_type).strip().lower(),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            quantized=bool(quantized),
            explain=bool(explain),
            debug_prompt=bool(debug_prompt),
            checkpoint_path=checkpoint_path,
            emb_dim=int(emb_dim),
            fallback_mode=str(fallback_mode),
            lightweight_mode=bool(lightweight_mode),
            phi4_model_id=str(phi4_model_id),
            phi4_precision=str(phi4_precision),
            phi4_estimated_vram_gb=float(phi4_estimated_vram_gb),
            vlm_model_id=str(vlm_model_id),
        )
        backend_obj = backend
        if backend_obj is None:
            btype = cfg.backend_type.strip().lower()
            if btype in {"stub", "tiny"}:
                if not cfg.lightweight_mode:
                    raise ValueError(
                        f"Backend '{btype}' requires lightweight_mode=True. "
                        "Default Phase-4 path requires a multimodal VLM backend."
                    )
                if btype == "stub":
                    backend_obj = StubBackend(config=cfg)
                    cfg.backend_type = "stub"
                else:
                    backend_obj = TinyLocalVLMBackend(config=cfg)
                    cfg.backend_type = "tiny"
            elif btype in {"phi4", "phi4_mm", "phi-4", "phi4-multimodal"}:
                backend_obj = Phi4MultimodalBackend(config=cfg)
                cfg.backend_type = "phi4_mm"
            elif btype in {"smolvlm", "smol_vlm", "hf_smolvlm"}:
                backend_obj = SmolVlmBackend(config=cfg)
                cfg.backend_type = "smolvlm"
            else:
                raise ValueError(
                    f"Unsupported hazard backend_type '{cfg.backend_type}'. "
                    "Use smolvlm or phi4_mm for multimodal VLMs, or tiny/stub with lightweight_mode=True."
                )
        super().__init__(backend=backend_obj, config=cfg)


class LaCHazardReasoner(DistilledHazardReasoner):
    """Legacy alias retained for compatibility."""

