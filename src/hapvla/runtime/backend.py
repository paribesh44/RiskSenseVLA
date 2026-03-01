"""Runtime backend selection with MPS-first defaults."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class BackendSpec:
    name: str
    device: str
    amp_dtype: torch.dtype | None


def pick_backend(preferred: str = "mps") -> BackendSpec:
    pref = preferred.lower()
    if pref == "mps" and torch.backends.mps.is_available():
        return BackendSpec(name="mps", device="mps", amp_dtype=torch.float16)
    if pref in {"cuda", "tensorrt"} and torch.cuda.is_available():
        return BackendSpec(name="cuda", device="cuda", amp_dtype=torch.float16)
    if torch.backends.mps.is_available():
        return BackendSpec(name="mps", device="mps", amp_dtype=torch.float16)
    if torch.cuda.is_available():
        return BackendSpec(name="cuda", device="cuda", amp_dtype=torch.float16)
    return BackendSpec(name="cpu", device="cpu", amp_dtype=None)
