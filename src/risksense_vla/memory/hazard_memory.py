"""Hazard-aware linear temporal memory."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from risksense_vla.types import Detection, HazardScore, MemoryObjectState, MemoryState


@dataclass(slots=True)
class HazardAwareMemory:
    """Risk-weighted memory where high-risk objects persist longer."""

    base_decay: float = 0.92
    min_persistence: float = 0.05
    max_persistence: float = 1.0
    emb_dim: int = 256
    state_dim: int = 512
    objects: dict[str, MemoryObjectState] = field(default_factory=dict)
    hoi_embedding: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 256))

    def _hazard_weight(self, label: str, hazards: list[HazardScore]) -> float:
        if not hazards:
            return 0.0
        scores = [h.score for h in hazards if h.object == label or h.subject == label]
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def update(
        self,
        timestamp: float,
        detections: list[Detection],
        embeddings: torch.Tensor,
        hazards: list[HazardScore],
    ) -> MemoryState:
        # Linear pass on active detections.
        seen: set[str] = set()
        for det in detections:
            seen.add(det.track_id)
            hz = self._hazard_weight(det.label, hazards)
            obj = self.objects.get(det.track_id)
            if obj is None:
                obj = MemoryObjectState(
                    track_id=det.track_id,
                    label=det.label,
                    last_bbox_xyxy=det.bbox_xyxy,
                    persistence=0.5 + 0.5 * hz,
                    hazard_weight=hz,
                    age_frames=1,
                )
            else:
                obj.last_bbox_xyxy = det.bbox_xyxy
                obj.age_frames += 1
                # High hazard -> slower decay and stronger persistence recovery.
                decay = self.base_decay + 0.06 * hz
                obj.persistence = min(self.max_persistence, obj.persistence * decay + 0.12 * hz)
                obj.hazard_weight = 0.7 * obj.hazard_weight + 0.3 * hz
            self.objects[det.track_id] = obj

        # Linear pass on stale objects with hazard-aware decay.
        stale_ids = [track_id for track_id in self.objects if track_id not in seen]
        for track_id in stale_ids:
            obj = self.objects[track_id]
            decay = self.base_decay + 0.05 * obj.hazard_weight
            obj.persistence *= decay
            if obj.persistence < self.min_persistence:
                del self.objects[track_id]
            else:
                self.objects[track_id] = obj

        if embeddings.numel() > 0:
            pooled = embeddings.mean(dim=0, keepdim=True)
            self.hoi_embedding = 0.8 * self.hoi_embedding + 0.2 * pooled
        state_vector = torch.zeros((1, self.state_dim), dtype=torch.float32)
        p_values = [o.persistence for o in self.objects.values()]
        h_values = [o.hazard_weight for o in self.objects.values()]
        state_vector[0, 0] = float(len(self.objects))
        state_vector[0, 1] = float(sum(p_values) / len(p_values)) if p_values else 0.0
        state_vector[0, 2] = float(sum(h_values) / len(h_values)) if h_values else 0.0
        copy_n = min(self.emb_dim, self.hoi_embedding.shape[-1])
        state_vector[0, 32 : 32 + copy_n] = self.hoi_embedding[0, :copy_n]
        return MemoryState(
            timestamp=timestamp,
            objects=list(self.objects.values()),
            hoi_embedding=self.hoi_embedding.clone(),
            state_vector=state_vector,
        )

    def summary(self) -> dict[str, float]:
        if not self.objects:
            return {"num_objects": 0.0, "avg_persistence": 0.0, "avg_hazard_weight": 0.0}
        objs = list(self.objects.values())
        return {
            "num_objects": float(len(objs)),
            "avg_persistence": float(sum(o.persistence for o in objs) / len(objs)),
            "avg_hazard_weight": float(sum(o.hazard_weight for o in objs) / len(objs)),
        }
