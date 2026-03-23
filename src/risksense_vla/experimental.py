"""Experiment controls for reproducibility, toggles, and perturbations."""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Any

import numpy as np
import torch

from risksense_vla.types import HOITriplet, PerceptionDetection

PAPER_METHOD_NAMES = {
    "hazard_aware": "HW-SSM (Proposed)",
    "naive": "No-Temporal-Memory",
    "frame_only": "Frame-Level HOI",
}


def seed_everything(seed: int) -> None:
    """Set RNG seeds across Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def get_bool(cfg: dict[str, Any], section: str, key: str, default: bool) -> bool:
    sec = cfg.get(section, {})
    if not isinstance(sec, dict):
        return default
    return bool(sec.get(key, default))


def resolve_mode(cfg: dict[str, Any], key: str, default: str) -> str:
    # Backward-compatible: support both baseline.* and baselines.*
    base_a = cfg.get("baseline", {})
    base_b = cfg.get("baselines", {})
    if isinstance(base_b, dict) and key in base_b:
        return str(base_b.get(key, default))
    if isinstance(base_a, dict) and key in base_a:
        return str(base_a.get(key, default))
    return default


def apply_occlusion(
    detections: list[PerceptionDetection],
    occlusion_prob: float,
    rng: random.Random,
) -> tuple[list[PerceptionDetection], list[dict[str, object]]]:
    """Randomly drop detections to simulate occlusion."""
    if occlusion_prob <= 0.0 or not detections:
        return detections, []
    kept: list[PerceptionDetection] = []
    events: list[dict[str, object]] = []
    for det in detections:
        sample = rng.random()
        if sample < occlusion_prob:
            events.append(
                {
                    "track_id": det.track_id,
                    "label": det.label,
                    "event": "dropped",
                    "sample": float(sample),
                    "occlusion_prob": float(occlusion_prob),
                }
            )
            continue
        kept.append(det)
    return kept, events


def top_observed_action(hois: list[HOITriplet]) -> str:
    observed = [h for h in hois if not h.predicted]
    if not observed:
        return ""
    best = max(observed, key=lambda h: float(h.confidence))
    return best.action


def top_predicted_actions_by_horizon(hois: list[HOITriplet], *, max_horizon: int = 3) -> dict[int, str]:
    grouped: dict[int, list[str]] = {}
    for hoi in hois:
        if not hoi.predicted:
            continue
        horizon = int(round(float(hoi.t_end - hoi.t_start)))
        if 1 <= horizon <= max_horizon:
            grouped.setdefault(horizon, []).append(hoi.action)
    out: dict[int, str] = {}
    for horizon, actions in grouped.items():
        if actions:
            out[horizon] = Counter(actions).most_common(1)[0][0]
    return out


def method_display_name(mode: str) -> str:
    """Map internal mode key to paper-facing method name."""
    return PAPER_METHOD_NAMES.get(mode, mode)

