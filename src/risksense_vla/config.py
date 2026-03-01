"""Config loading and typed accessors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RuntimeConfig:
    backend: str
    device: str
    mixed_precision: bool
    quant_bits: int
    pruning_ratio: float
    semantic_attention_threshold: float


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_config(default_path: str | Path, override_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_yaml(default_path)
    if override_path:
        cfg = merge_dicts(cfg, load_yaml(override_path))
    return cfg


def runtime_config(cfg: dict[str, Any]) -> RuntimeConfig:
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
