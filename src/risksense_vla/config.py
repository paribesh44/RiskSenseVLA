"""Config loading, validation, and typed accessors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_LOG = logging.getLogger(__name__)

_REQUIRED_SECTIONS = ("runtime", "perception", "hazard")

_SECTION_SCHEMA: dict[str, dict[str, type]] = {
    "runtime": {
        "backend": str,
        "device": str,
        "mixed_precision": bool,
        "target_fps": (int, float),  # type: ignore[dict-item]
    },
    "perception": {
        "detector_backend": str,
        "embedding_dim": int,
        "detector_confidence_threshold": (int, float),  # type: ignore[dict-item]
    },
    "hazard": {
        "alert_threshold": (int, float),  # type: ignore[dict-item]
        "backend_type": str,
        "use_vlm": bool,
    },
    "memory": {"use_hazard_weighting": bool},
    "hoi": {"use_prediction": bool},
    "evaluation": {
        "occlusion_prob": (int, float),  # type: ignore[dict-item]
        "occlusion_levels": list,
    },
    "reproducibility": {"seed": int},
    "optimization": {
        "quant_bits": int,
    },
    "attention": {
        "semantic_attention_threshold": (int, float),  # type: ignore[dict-item]
    },
}


@dataclass(slots=True)
class RuntimeConfig:
    """Typed runtime configuration extracted from the raw config dict."""

    backend: str
    device: str
    mixed_precision: bool
    quant_bits: int
    pruning_ratio: float
    semantic_attention_threshold: float


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Validate a loaded config dict against the expected schema.

    Returns a list of warning/error messages.  An empty list means the
    config is valid.
    """
    issues: list[str] = []
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            issues.append(f"missing required section '{section}'")

    for section, fields in _SECTION_SCHEMA.items():
        sec_data = cfg.get(section, {})
        if not isinstance(sec_data, dict):
            issues.append(f"section '{section}' should be a mapping, got {type(sec_data).__name__}")
            continue
        for key, expected_type in fields.items():
            if key not in sec_data:
                continue
            val = sec_data[key]
            if not isinstance(val, expected_type):
                issues.append(
                    f"{section}.{key}: expected {expected_type}, got {type(val).__name__} ({val!r})"
                )

    emb_dim = cfg.get("perception", {}).get("embedding_dim", 256)
    if isinstance(emb_dim, int) and emb_dim <= 0:
        issues.append(f"perception.embedding_dim must be positive, got {emb_dim}")

    threshold = cfg.get("hazard", {}).get("alert_threshold", 0.65)
    if isinstance(threshold, (int, float)) and not (0.0 <= threshold <= 1.0):
        issues.append(f"hazard.alert_threshold must be in [0, 1], got {threshold}")

    att_thresh = cfg.get("attention", {}).get("semantic_attention_threshold", 0.6)
    if isinstance(att_thresh, (int, float)) and not (0.0 <= att_thresh <= 1.0):
        issues.append(f"attention.semantic_attention_threshold must be in [0, 1], got {att_thresh}")

    occ_prob = cfg.get("evaluation", {}).get("occlusion_prob", 0.0)
    if isinstance(occ_prob, (int, float)) and not (0.0 <= occ_prob <= 1.0):
        issues.append(f"evaluation.occlusion_prob must be in [0, 1], got {occ_prob}")
    occ_levels = cfg.get("evaluation", {}).get("occlusion_levels", [])
    if occ_levels:
        if not isinstance(occ_levels, list):
            issues.append("evaluation.occlusion_levels must be a list of numbers in [0, 1]")
        else:
            for idx, level in enumerate(occ_levels):
                if not isinstance(level, (int, float)) or not (0.0 <= float(level) <= 1.0):
                    issues.append(
                        f"evaluation.occlusion_levels[{idx}] must be numeric in [0, 1], got {level!r}"
                    )

    for issue in issues:
        _LOG.warning("config validation: %s", issue)

    return issues


def load_config(default_path: str | Path, override_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default config and optionally merge an override config."""
    cfg = load_yaml(default_path)
    if override_path:
        cfg = merge_dicts(cfg, load_yaml(override_path))
    return cfg


def runtime_config(cfg: dict[str, Any]) -> RuntimeConfig:
    """Extract a typed RuntimeConfig from the raw config dict."""
    r = cfg.get("runtime", {})
    q = cfg.get("optimization", {})
    a = cfg.get("attention", {})
    return RuntimeConfig(
        backend=str(r.get("backend", "mps")),
        device=str(r.get("device", "auto")),
        mixed_precision=bool(r.get("mixed_precision", True)),
        quant_bits=int(q.get("quant_bits", 8)),
        pruning_ratio=float(q.get("pruning_ratio", 0.0)),
        semantic_attention_threshold=float(a.get("semantic_attention_threshold", 0.6)),
    )
