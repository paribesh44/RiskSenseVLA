"""Dataset adapters for predictive HOI training."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

_LOG = logging.getLogger(__name__)


def _text_embedding(text: str, emb_dim: int) -> torch.Tensor:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator()
    g.manual_seed(seed)
    emb = torch.randn((emb_dim,), generator=g, dtype=torch.float32)
    return emb / (torch.linalg.norm(emb) + 1e-8)


def _ensure_list(payload: Any) -> list[dict[str, Any]]:  # payload: JSON-loaded structure
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("samples", "annotations", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _safe_int(value: Any, default: int) -> int:  # value from JSON parsing
    try:
        return int(value)
    except Exception as exc:
        _LOG.debug("_safe_int failed for %r: %s", value, exc)
        return default


def _safe_str(value: Any, default: str) -> str:  # value from JSON parsing
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _fit_embedding(raw: Any, emb_dim: int) -> torch.Tensor:  # raw from JSON (list/tensor/None)
    out = torch.zeros((emb_dim,), dtype=torch.float32)
    if raw is None:
        return out
    tensor = torch.as_tensor(raw, dtype=torch.float32).flatten()
    n = min(emb_dim, int(tensor.shape[0]))
    if n > 0:
        out[:n] = tensor[:n]
    return out


@dataclass(slots=True)
class HOIEvent:
    video_id: str
    frame_idx: int
    action: str
    object_label: str
    subject: str = "human"
    object_embedding: torch.Tensor | None = None
    memory_embedding: torch.Tensor | None = None


class _BaseTemporalHOIDataset(Dataset):
    def __init__(
        self,
        events: list[HOIEvent],
        *,
        action_vocab: list[str],
        emb_dim: int = 256,
        horizon_seconds: int = 3,
    ) -> None:
        self.events = sorted(events, key=lambda x: (x.video_id, x.frame_idx))
        self.emb_dim = emb_dim
        self.horizon_seconds = max(1, min(3, int(horizon_seconds)))
        self.action_vocab = action_vocab
        self.action_to_idx = {action: i for i, action in enumerate(action_vocab)}
        self._group_indices: dict[str, list[int]] = {}
        for i, ev in enumerate(self.events):
            self._group_indices.setdefault(ev.video_id, []).append(i)

    def __len__(self) -> int:
        return len(self.events)

    def _action_index(self, action: str) -> int:
        return self.action_to_idx.get(action, 0)

    def _future_events(self, idx: int) -> list[HOIEvent]:
        current = self.events[idx]
        group = self._group_indices.get(current.video_id, [])
        if not group:
            return []
        pos = group.index(idx)
        out: list[HOIEvent] = []
        for j in range(pos + 1, min(len(group), pos + 1 + self.horizon_seconds)):
            out.append(self.events[group[j]])
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ev = self.events[idx]
        future = self._future_events(idx)
        obj_emb = ev.object_embedding if ev.object_embedding is not None else _text_embedding(ev.object_label, self.emb_dim)
        mem_emb = ev.memory_embedding if ev.memory_embedding is not None else _text_embedding(
            f"{ev.subject}:{ev.action}", self.emb_dim
        )
        current_idx = self._action_index(ev.action)

        future_indices = torch.full((self.horizon_seconds,), current_idx, dtype=torch.long)
        future_embs = torch.zeros((self.horizon_seconds, self.emb_dim), dtype=torch.float32)
        for step in range(self.horizon_seconds):
            if step < len(future):
                f_ev = future[step]
                future_indices[step] = self._action_index(f_ev.action)
                base = f_ev.object_embedding if f_ev.object_embedding is not None else _text_embedding(
                    f"{f_ev.action}:{f_ev.object_label}", self.emb_dim
                )
                future_embs[step] = base
            else:
                future_embs[step] = _text_embedding(f"{ev.action}:{ev.object_label}:{step}", self.emb_dim)

        return {
            "object_embedding": obj_emb.to(torch.float32),
            "memory_embedding": mem_emb.to(torch.float32),
            "current_action_idx": torch.tensor(current_idx, dtype=torch.long),
            "future_action_indices": future_indices,
            "future_embeddings": future_embs,
        }


class HOIGenRawDataset(_BaseTemporalHOIDataset):
    """Raw HOIGen-style adapter from annotation JSON."""

    def __init__(
        self,
        annotation_json: str | Path,
        *,
        action_vocab: list[str],
        emb_dim: int = 256,
        horizon_seconds: int = 3,
    ) -> None:
        payload = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
        events: list[HOIEvent] = []
        for sample in _ensure_list(payload):
            video_id = _safe_str(sample.get("video_id"), "unknown")
            frames = sample.get("frames", [])
            if not isinstance(frames, list):
                continue
            for frame_idx, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    continue
                frame_id = _safe_int(frame.get("frame_idx"), frame_idx)
                for hoi in frame.get("hois", []):
                    if not isinstance(hoi, dict):
                        continue
                    action = _safe_str(hoi.get("action"), "interact")
                    obj = _safe_str(hoi.get("object"), "object")
                    subj = _safe_str(hoi.get("subject"), "human")
                    events.append(
                        HOIEvent(
                            video_id=video_id,
                            frame_idx=frame_id,
                            action=action,
                            object_label=obj,
                            subject=subj,
                            object_embedding=_fit_embedding(hoi.get("object_embedding"), emb_dim),
                            memory_embedding=_fit_embedding(hoi.get("memory_embedding"), emb_dim),
                        )
                    )
        super().__init__(events, action_vocab=action_vocab, emb_dim=emb_dim, horizon_seconds=horizon_seconds)


class HICODetRawDataset(_BaseTemporalHOIDataset):
    """Raw HICO-DET style adapter from annotation JSON."""

    def __init__(
        self,
        annotation_json: str | Path,
        *,
        action_vocab: list[str],
        emb_dim: int = 256,
        horizon_seconds: int = 3,
    ) -> None:
        payload = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
        events: list[HOIEvent] = []
        for idx, item in enumerate(_ensure_list(payload)):
            image_id = _safe_str(item.get("image_id"), f"image_{idx}")
            # HICO-DET is typically image-level. We treat each image as frame_idx 0 and
            # use repeated future labels for compatibility with predictive training.
            interactions = item.get("hois")
            if not isinstance(interactions, list):
                interactions = item.get("hoi_annotation", [])
            for hoi in interactions:
                if not isinstance(hoi, dict):
                    continue
                action = _safe_str(hoi.get("action"), hoi.get("verb"))
                action = action if action else "interact"
                obj = _safe_str(hoi.get("object"), hoi.get("object_name"))
                obj = obj if obj else "object"
                events.append(
                    HOIEvent(
                        video_id=image_id,
                        frame_idx=0,
                        action=action,
                        object_label=obj,
                        subject="human",
                        object_embedding=_fit_embedding(hoi.get("object_embedding"), emb_dim),
                        memory_embedding=_fit_embedding(hoi.get("memory_embedding"), emb_dim),
                    )
                )
        super().__init__(events, action_vocab=action_vocab, emb_dim=emb_dim, horizon_seconds=horizon_seconds)


class TemporalHOIPreprocessedDataset(_BaseTemporalHOIDataset):
    """Loader for JSONL generated by preprocess_hoi-style scripts."""

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        action_vocab: list[str],
        emb_dim: int = 256,
        horizon_seconds: int = 3,
    ) -> None:
        events: list[HOIEvent] = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessed JSONL: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            video_id = _safe_str(rec.get("video_id"), "unknown")
            hois = rec.get("hois", [])
            if not isinstance(hois, list):
                continue
            for hoi in hois:
                if not isinstance(hoi, dict):
                    continue
                events.append(
                    HOIEvent(
                        video_id=video_id,
                        frame_idx=_safe_int(hoi.get("frame_idx"), _safe_int(rec.get("start_frame"), 0)),
                        action=_safe_str(hoi.get("action"), "interact"),
                        object_label=_safe_str(hoi.get("object"), "object"),
                        subject=_safe_str(hoi.get("subject"), "human"),
                        object_embedding=_fit_embedding(hoi.get("object_embedding"), emb_dim),
                        memory_embedding=_fit_embedding(hoi.get("memory_embedding"), emb_dim),
                    )
                )
        super().__init__(events, action_vocab=action_vocab, emb_dim=emb_dim, horizon_seconds=horizon_seconds)


def build_hoi_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )

