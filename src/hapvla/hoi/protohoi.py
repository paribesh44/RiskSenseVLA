"""ProtoHOI-style zero-shot and predictive HOI reasoning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict

import torch

from hapvla.types import Detection, HOITriplet, MemoryState


def _text_proto(text: str, dim: int = 256) -> torch.Tensor:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn((dim,), generator=g)
    return v / (torch.linalg.norm(v) + 1e-8)


@dataclass(slots=True)
class ProtoHOIPredictor:
    """Simple prototype-based HOI predictor with temporal anticipation."""

    future_horizon_seconds: int = 3
    emb_dim: int = 256
    actions: list[str] = field(
        default_factory=lambda: ["hold", "cut", "pour", "open", "touch_hot_surface", "carry", "drop"]
    )
    action_prototypes: Dict[str, torch.Tensor] = field(init=False)

    def __post_init__(self) -> None:
        self.action_prototypes = {a: _text_proto(a, self.emb_dim) for a in self.actions}

    def _best_action(self, emb: torch.Tensor) -> tuple[str, float]:
        scores = {}
        emb = emb / (torch.linalg.norm(emb) + 1e-8)
        for action, proto in self.action_prototypes.items():
            scores[action] = float(torch.dot(emb, proto))
        best = max(scores.items(), key=lambda x: x[1])
        conf = (best[1] + 1.0) / 2.0
        return best[0], conf

    def predict(
        self,
        timestamp: float,
        detections: list[Detection],
        embeddings: torch.Tensor,
        memory: MemoryState,
    ) -> list[HOITriplet]:
        if not detections or embeddings.numel() == 0:
            return []
        triplets: list[HOITriplet] = []
        subject = "human"
        mem_bias = memory.hoi_embedding[0, : self.emb_dim]
        for det in detections:
            idx = max(0, det.embedding_idx)
            obj_emb = embeddings[idx]
            action, conf = self._best_action(0.7 * obj_emb + 0.3 * mem_bias)
            triplets.append(
                HOITriplet(
                    subject=subject,
                    action=action,
                    object=det.label,
                    confidence=min(1.0, conf * det.confidence),
                    t_start=timestamp,
                    t_end=timestamp,
                    predicted=False,
                )
            )
            for dt in range(1, self.future_horizon_seconds + 1):
                future_action, future_conf = self._best_action(obj_emb + (0.15 * dt) * mem_bias)
                triplets.append(
                    HOITriplet(
                        subject=subject,
                        action=future_action,
                        object=det.label,
                        confidence=min(1.0, future_conf * 0.85 * det.confidence),
                        t_start=timestamp,
                        t_end=timestamp + float(dt),
                        predicted=True,
                    )
                )
        return triplets
