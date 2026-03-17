"""Hazard-aware temporal memory with a lightweight linear SSM core.

Linear SSM used here
--------------------
Let:
  - ``u_t`` be the frame input built from detections/embeddings.
  - ``x_t`` be the latent SSM state (``self.ssm_state``).
  - ``W_in`` be ``self._input_proj`` and ``W_out`` be ``self._output_proj``.
  - ``alpha`` be ``self.ssm_alpha`` and ``beta`` be ``self.ssm_beta``.
  - ``g_t = 1 + 0.35 * avg_hazard_t``.

The implemented recurrence is:
  ``x_t = alpha * x_{t-1} + beta * g_t * (u_t @ W_in)``

and the emitted temporal embedding is:
  ``e_t = normalize(x_t @ W_out)``
  ``hoi_t = (1 - mix_t) * hoi_{t-1} + mix_t * e_t``
with ``mix_t = clamp(embedding_mix + 0.2 * avg_hazard_t, 0, 0.8)``.

Where hazard enters the system
------------------------------
1) Input aggregation weighting (in ``_build_frame_input``):
   ``weight_i = 0.5 + 0.5 * hazard_i`` for each detection.
2) SSM input gate (in ``_update_ssm``):
   ``g_t`` scales the input drive term ``B u_t``.
3) Object persistence dynamics (in ``update``):
   - observed object:
     ``p_t = clip(p_{t-1} * (base_decay + hazard_retention_gain * h_t) + observation_boost * (0.5 + 0.5 * h_t))``
   - stale object:
     ``p_t = clip(p_{t-1} * (base_decay - stale_decay_penalty + hazard_retention_gain * h_prev))``

Why this is a linear SSM
------------------------
The latent transition is linear in previous state and input
(``x_t = A_t x_{t-1} + B_t u_t``), where:
  - ``A_t = alpha * I``
  - ``B_t = beta * g_t * W_in``
``g_t`` is an exogenous scalar from hazard signals, making this a
linear time-varying SSM rather than a nonlinear recurrent model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch

from risksense_vla.types import HazardScore, MemoryObjectState, MemoryState, PerceptionDetection

_LOGGER = logging.getLogger(__name__)
_STATE_OFFSET = 32


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to the given bounds [low, high]."""
    return max(low, min(high, value))


def _copy_object_state(obj: MemoryObjectState) -> MemoryObjectState:
    """Return a deep copy of a MemoryObjectState for safe state snapshots."""
    return MemoryObjectState(
        track_id=obj.track_id,
        label=obj.label,
        last_bbox_xyxy=tuple(obj.last_bbox_xyxy),
        persistence=float(obj.persistence),
        hazard_weight=float(obj.hazard_weight),
        age_frames=int(obj.age_frames),
    )


@dataclass(slots=True)
class HazardAwareMemory:
    """Linear-time hazard-aware memory with SSM state for temporal context."""

    base_decay: float = 0.86
    stale_decay_penalty: float = 0.08
    hazard_retention_gain: float = 0.14
    observation_boost: float = 0.20
    min_persistence: float = 0.05
    max_persistence: float = 1.0
    emb_dim: int = 256
    state_dim: int = 512
    ssm_state_dim: int = 128
    ssm_alpha: float = 0.90
    ssm_beta: float = 0.20
    embedding_mix: float = 0.18
    log_updates: bool = False
    objects: dict[str, MemoryObjectState] = field(default_factory=dict)
    hoi_embedding: torch.Tensor = field(default_factory=lambda: torch.zeros((1, 256), dtype=torch.float32))
    ssm_state: torch.Tensor = field(default_factory=lambda: torch.zeros((1, 128), dtype=torch.float32))
    _input_proj: torch.Tensor = field(init=False, repr=False)
    _output_proj: torch.Tensor = field(init=False, repr=False)
    _input_dim: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.hoi_embedding = self._fit_embedding(self.hoi_embedding)
        self.ssm_state = self._fit_ssm_state(self.ssm_state)
        self._input_dim = self.emb_dim + 8
        self._input_proj = self._make_projection(self._input_dim, self.ssm_state_dim, seed=17)
        self._output_proj = self._make_projection(self.ssm_state_dim, self.emb_dim, seed=31)

    @classmethod
    def from_memory_state(
        cls,
        previous_memory_state: MemoryState | None,
        **kwargs: object,
    ) -> HazardAwareMemory:
        memory = cls(**kwargs)
        if previous_memory_state is None:
            return memory
        memory._load_previous_state(previous_memory_state)
        return memory

    def _make_projection(self, in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(seed)
        scale = 1.0 / max(1, in_dim) ** 0.5
        return torch.randn((in_dim, out_dim), generator=generator, dtype=torch.float32) * scale

    def _fit_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        out = torch.zeros((1, self.emb_dim), dtype=torch.float32)
        if embedding.numel() == 0:
            return out
        copy_n = min(self.emb_dim, embedding.shape[-1])
        out[0, :copy_n] = embedding.to(torch.float32)[0, :copy_n]
        return out

    def _fit_ssm_state(self, ssm_state: torch.Tensor) -> torch.Tensor:
        if ssm_state.ndim == 1:
            ssm_state = ssm_state.unsqueeze(0)
        out = torch.zeros((1, self.ssm_state_dim), dtype=torch.float32)
        if ssm_state.numel() == 0:
            return out
        copy_n = min(self.ssm_state_dim, ssm_state.shape[-1])
        out[0, :copy_n] = ssm_state.to(torch.float32)[0, :copy_n]
        return out

    def _load_previous_state(self, previous_memory_state: MemoryState) -> None:
        self.objects = {obj.track_id: _copy_object_state(obj) for obj in previous_memory_state.objects}
        self.hoi_embedding = self._fit_embedding(previous_memory_state.hoi_embedding)
        prev_state = previous_memory_state.state_vector
        if prev_state.ndim == 1:
            prev_state = prev_state.unsqueeze(0)
        start = _STATE_OFFSET + self.emb_dim
        extracted = torch.zeros((1, self.ssm_state_dim), dtype=torch.float32)
        if prev_state.numel() > 0 and prev_state.shape[-1] > start:
            copy_n = min(self.ssm_state_dim, prev_state.shape[-1] - start)
            extracted[0, :copy_n] = prev_state.to(torch.float32)[0, start : start + copy_n]
        self.ssm_state = extracted

    def _hazard_weight(self, label: str, hazards: list[HazardScore]) -> float:
        if not hazards:
            return 0.0
        scores = [h.score for h in hazards if h.object == label or h.subject == label]
        if not scores:
            return 0.0
        return _clamp(float(sum(scores) / len(scores)), 0.0, 1.0)

    def _resolve_hazard_score(
        self,
        det_idx: int,
        det: PerceptionDetection,
        hazards: list[float] | None,
        hazard_events: list[HazardScore],
    ) -> float:
        if hazards is not None and det_idx < len(hazards):
            return _clamp(float(hazards[det_idx]), 0.0, 1.0)
        if hazard_events:
            return self._hazard_weight(det.label, hazard_events)
        return 1.0

    def _embedding_for_detection(
        self,
        det: PerceptionDetection,
    ) -> torch.Tensor:
        if det.clip_embedding.numel() == 0:
            return torch.zeros((1, self.emb_dim), dtype=torch.float32)
        out = torch.zeros((1, self.emb_dim), dtype=torch.float32)
        emb = det.clip_embedding.to(torch.float32).flatten()
        copy_n = min(self.emb_dim, emb.shape[-1])
        out[0, :copy_n] = emb[:copy_n]
        return out

    def _build_frame_input(
        self,
        detections: list[PerceptionDetection],
        resolved_hazards: list[float],
    ) -> tuple[torch.Tensor, float]:
        frame_input = torch.zeros((1, self._input_dim), dtype=torch.float32)
        if not detections:
            return frame_input, 0.0
        weighted_sum = torch.zeros((1, self.emb_dim + 5), dtype=torch.float32)
        weight_total = 0.0
        conf_sum = 0.0
        hz_sum = 0.0
        for i, det in enumerate(detections):
            hz = resolved_hazards[i]
            emb = self._embedding_for_detection(det)
            x1, y1, x2, y2 = det.bbox_xyxy
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            box_feat = torch.tensor(
                [[cx / 1000.0, cy / 1000.0, w / 1000.0, h / 1000.0, float(det.confidence)]],
                dtype=torch.float32,
            )
            det_vec = torch.cat((emb, box_feat), dim=1)
            weight = 0.5 + 0.5 * hz
            weighted_sum += det_vec * weight
            weight_total += weight
            conf_sum += float(det.confidence)
            hz_sum += hz
        pooled = weighted_sum / max(weight_total, 1e-6)
        frame_input[0, : self.emb_dim + 5] = pooled[0]
        avg_hazard = hz_sum / max(1, len(detections))
        frame_input[0, self.emb_dim + 5] = min(1.0, len(detections) / 32.0)
        frame_input[0, self.emb_dim + 6] = avg_hazard
        frame_input[0, self.emb_dim + 7] = conf_sum / max(1, len(detections))
        return frame_input, avg_hazard

    def _update_ssm(self, frame_input: torch.Tensor, avg_hazard: float) -> None:
        """Apply one linear-SSM step with hazard-conditioned input gain."""
        projected_input = frame_input @ self._input_proj
        hazard_gate = 1.0 + 0.35 * avg_hazard
        self.ssm_state = self.ssm_alpha * self.ssm_state + self.ssm_beta * hazard_gate * projected_input
        ssm_out = self.ssm_state @ self._output_proj
        ssm_out = ssm_out / (torch.linalg.norm(ssm_out, dim=-1, keepdim=True) + 1e-6)
        mix = _clamp(self.embedding_mix + 0.2 * avg_hazard, 0.0, 0.8)
        self.hoi_embedding = (1.0 - mix) * self.hoi_embedding + mix * ssm_out

    def _build_state_vector(
        self,
        observed_count: int,
        stale_count: int,
    ) -> torch.Tensor:
        state_vector = torch.zeros((1, self.state_dim), dtype=torch.float32)
        objs = list(self.objects.values())
        p_values = [o.persistence for o in objs]
        h_values = [o.hazard_weight for o in objs]
        age_values = [o.age_frames for o in objs]
        state_vector[0, 0] = float(len(objs))
        state_vector[0, 1] = float(sum(p_values) / len(p_values)) if p_values else 0.0
        state_vector[0, 2] = float(sum(h_values) / len(h_values)) if h_values else 0.0
        state_vector[0, 3] = float(observed_count)
        state_vector[0, 4] = float(stale_count)
        state_vector[0, 5] = float(sum(age_values) / len(age_values)) if age_values else 0.0
        state_vector[0, 6] = float(torch.linalg.norm(self.ssm_state).item())
        copy_emb = min(self.emb_dim, max(0, self.state_dim - _STATE_OFFSET))
        state_vector[0, _STATE_OFFSET : _STATE_OFFSET + copy_emb] = self.hoi_embedding[0, :copy_emb]
        ssm_offset = _STATE_OFFSET + self.emb_dim
        if ssm_offset < self.state_dim:
            copy_ssm = min(self.ssm_state_dim, self.state_dim - ssm_offset)
            state_vector[0, ssm_offset : ssm_offset + copy_ssm] = self.ssm_state[0, :copy_ssm]
        return state_vector

    def _maybe_log(
        self,
        timestamp: float,
        observed_count: int,
        stale_count: int,
        callback: Callable[[dict[str, float]], None] | None,
    ) -> None:
        summary = self.summary()
        payload = {
            "timestamp": float(timestamp),
            "num_objects": summary["num_objects"],
            "avg_persistence": summary["avg_persistence"],
            "avg_hazard_weight": summary["avg_hazard_weight"],
            "state_norm": float(torch.linalg.norm(self.ssm_state).item()),
            "observed_count": float(observed_count),
            "stale_count": float(stale_count),
        }
        if callback is not None:
            callback(payload)
            return
        if self.log_updates:
            _LOGGER.info("memory_update %s", payload)

    def update(
        self,
        timestamp: float,
        detections: list[PerceptionDetection],
        hazards: list[float] | None = None,
        hazard_events: list[HazardScore] | None = None,
        previous_memory_state: MemoryState | None = None,
        log_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> MemoryState:
        if previous_memory_state is not None:
            self._load_previous_state(previous_memory_state)
        hazard_events = hazard_events or []

        seen: set[str] = set()
        resolved_hazards: list[float] = []
        for i, det in enumerate(detections):
            seen.add(det.track_id)
            hz = self._resolve_hazard_score(i, det, hazards, hazard_events)
            resolved_hazards.append(hz)
            obj = self.objects.get(det.track_id)
            if obj is None:
                obj = MemoryObjectState(
                    track_id=det.track_id,
                    label=det.label,
                    last_bbox_xyxy=det.bbox_xyxy,
                    persistence=_clamp(0.35 + 0.60 * hz, self.min_persistence, self.max_persistence),
                    hazard_weight=hz,
                    age_frames=1,
                )
            else:
                obj.last_bbox_xyxy = det.bbox_xyxy
                obj.age_frames += 1
                retain = _clamp(self.base_decay + self.hazard_retention_gain * hz, 0.0, 0.995)
                recover = self.observation_boost * (0.5 + 0.5 * hz)
                obj.persistence = _clamp(
                    obj.persistence * retain + recover,
                    self.min_persistence,
                    self.max_persistence,
                )
                obj.hazard_weight = _clamp(0.75 * obj.hazard_weight + 0.25 * hz, 0.0, 1.0)
            self.objects[det.track_id] = obj

        stale_count = 0
        for track_id in tuple(self.objects):
            if track_id in seen:
                continue
            stale_count += 1
            obj = self.objects[track_id]
            retain = _clamp(
                self.base_decay - self.stale_decay_penalty + self.hazard_retention_gain * obj.hazard_weight,
                0.0,
                0.995,
            )
            obj.persistence = _clamp(obj.persistence * retain, 0.0, self.max_persistence)
            if obj.persistence < self.min_persistence:
                del self.objects[track_id]
            else:
                self.objects[track_id] = obj

        frame_input, avg_hazard = self._build_frame_input(detections, resolved_hazards)
        if not detections and self.objects:
            avg_hazard = float(sum(o.hazard_weight for o in self.objects.values()) / len(self.objects))
        self._update_ssm(frame_input=frame_input, avg_hazard=avg_hazard)

        state_vector = self._build_state_vector(observed_count=len(seen), stale_count=stale_count)
        memory_state = MemoryState(
            timestamp=timestamp,
            objects=[_copy_object_state(obj) for obj in sorted(self.objects.values(), key=lambda x: x.track_id)],
            hoi_embedding=self.hoi_embedding.clone(),
            state_vector=state_vector,
        )
        self._maybe_log(
            timestamp=timestamp,
            observed_count=len(seen),
            stale_count=stale_count,
            callback=log_callback,
        )
        return memory_state

    def summary(self) -> dict[str, float]:
        if not self.objects:
            return {"num_objects": 0.0, "avg_persistence": 0.0, "avg_hazard_weight": 0.0}
        objs = list(self.objects.values())
        return {
            "num_objects": float(len(objs)),
            "avg_persistence": float(sum(o.persistence for o in objs) / len(objs)),
            "avg_hazard_weight": float(sum(o.hazard_weight for o in objs) / len(objs)),
        }


def update_hazard_memory(
    *,
    timestamp: float,
    detections: list[PerceptionDetection],
    previous_memory_state: MemoryState | None,
    hazards: list[float] | None = None,
    hazard_events: list[HazardScore] | None = None,
    log_callback: Callable[[dict[str, float]], None] | None = None,
    memory_kwargs: dict[str, object] | None = None,
) -> MemoryState:
    """Functional memory update API for explicit state in/out workflows."""
    memory = HazardAwareMemory.from_memory_state(previous_memory_state, **(memory_kwargs or {}))
    return memory.update(
        timestamp=timestamp,
        detections=detections,
        hazards=hazards,
        hazard_events=hazard_events,
        log_callback=log_callback,
    )
