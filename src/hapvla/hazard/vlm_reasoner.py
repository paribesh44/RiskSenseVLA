"""Language-as-Cost hazard scoring with lightweight VLM-style priors."""

from __future__ import annotations

from dataclasses import dataclass, field

from hapvla.types import HOITriplet, HazardScore


@dataclass(slots=True)
class HazardReasoningOutput:
    hazards: list[HazardScore]
    alerts: list[str]
    hazard_map: dict[str, float]


@dataclass(slots=True)
class LaCHazardReasoner:
    """Small-footprint hazard reasoner with rule priors and NL alerts."""

    alert_threshold: float = 0.65
    action_risk: dict[str, float] = field(
        default_factory=lambda: {
            "touch_hot_surface": 0.95,
            "cut": 0.8,
            "drop": 0.7,
            "pour": 0.55,
            "open": 0.4,
            "hold": 0.3,
            "carry": 0.45,
        }
    )
    object_risk: dict[str, float] = field(
        default_factory=lambda: {
            "knife": 0.95,
            "stove": 0.9,
            "vehicle": 0.9,
            "glass": 0.55,
            "person": 0.2,
            "bottle": 0.2,
        }
    )

    def _severity(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    def infer(self, hois: list[HOITriplet]) -> HazardReasoningOutput:
        hazards: list[HazardScore] = []
        alerts: list[str] = []
        hazard_map: dict[str, float] = {}
        for hoi in hois:
            a = self.action_risk.get(hoi.action, 0.35)
            o = self.object_risk.get(hoi.object, 0.3)
            pred_penalty = 0.9 if hoi.predicted else 1.0
            score = min(1.0, (0.6 * a + 0.4 * o) * hoi.confidence * pred_penalty + 0.1 * a)
            sev = self._severity(score)
            item = HazardScore(
                subject=hoi.subject,
                action=hoi.action,
                object=hoi.object,
                score=score,
                severity=sev,
                explanation=f"LaC risk estimate from action({a:.2f}) + object({o:.2f})",
            )
            hazards.append(item)
            key = f"{hoi.subject}:{hoi.action}:{hoi.object}"
            hazard_map[key] = score
            if score >= self.alert_threshold:
                horizon = "future" if hoi.predicted else "current"
                alerts.append(
                    f"[{sev.upper()}] {horizon} risk: {hoi.subject} may {hoi.action} {hoi.object} (score={score:.2f})"
                )
        return HazardReasoningOutput(hazards=hazards, alerts=alerts, hazard_map=hazard_map)
