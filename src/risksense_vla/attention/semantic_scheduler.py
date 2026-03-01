"""Semantic-aware compute allocation using hazard signals."""

from __future__ import annotations

from dataclasses import dataclass

from risksense_vla.types import Detection, HazardScore


@dataclass(slots=True)
class SemanticAttentionScheduler:
    threshold: float = 0.6
    low_risk_scale: float = 0.5
    high_risk_scale: float = 1.0

    def _risk_for_label(self, label: str, hazards: list[HazardScore]) -> float:
        scores = [h.score for h in hazards if h.object == label or h.subject == label]
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def allocation(self, detections: list[Detection], hazards: list[HazardScore]) -> dict[str, float]:
        out: dict[str, float] = {}
        for det in detections:
            risk = self._risk_for_label(det.label, hazards)
            out[det.track_id] = self.high_risk_scale if risk >= self.threshold else self.low_risk_scale
        return out
