"""Predictive HOI module with zero-shot inference and train/fine-tune routines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from risksense_vla.types import HOITriplet, MemoryState, PerceptionDetection


def _text_proto(text: str, dim: int) -> torch.Tensor:
    """Build a deterministic L2-normalized embedding from text via SHA256 hash seed."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator()
    g.manual_seed(seed)
    vec = torch.randn((dim,), generator=g, dtype=torch.float32)
    return vec / (torch.linalg.norm(vec) + 1e-8)


def _normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a vector (or vectors along last dim) with optional epsilon for stability."""
    if vec.ndim == 1:
        return vec / (torch.linalg.norm(vec) + eps)
    return vec / (torch.linalg.norm(vec, dim=-1, keepdim=True) + eps)


def _fit_embedding(emb: torch.Tensor | None, emb_dim: int) -> torch.Tensor:
    out = torch.zeros((emb_dim,), dtype=torch.float32)
    if emb is None or emb.numel() == 0:
        return out
    emb = emb.detach().to(torch.float32).flatten()
    n = min(emb_dim, emb.shape[0])
    out[:n] = emb[:n]
    return out


def _to_device(x: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if x is None:
        return None
    return x.to(device)


@dataclass(slots=True)
class HOI:
    subject: str
    action: str
    object: str
    confidence: float
    t_start: float
    t_end: float


@dataclass(slots=True)
class HOIInferenceOutput:
    hoi_current: list[HOI]
    hoi_future_embeddings: torch.Tensor
    future_action_labels: list[list[str]] = field(default_factory=list)
    future_action_confidences: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0), dtype=torch.float32))

    def as_triplets(self) -> list[HOITriplet]:
        """Convert current/future outputs to legacy HOI triplets for downstream modules."""
        triplets: list[HOITriplet] = []
        for idx, hoi in enumerate(self.hoi_current):
            triplets.append(
                HOITriplet(
                    subject=hoi.subject,
                    action=hoi.action,
                    object=hoi.object,
                    confidence=float(hoi.confidence),
                    t_start=float(hoi.t_start),
                    t_end=float(hoi.t_end),
                    predicted=False,
                )
            )
            if idx >= len(self.future_action_labels):
                continue
            conf_row = self.future_action_confidences[idx] if idx < self.future_action_confidences.shape[0] else None
            for step, action in enumerate(self.future_action_labels[idx], start=1):
                conf = float(conf_row[step - 1].item()) if conf_row is not None and step - 1 < conf_row.shape[0] else 0.0
                triplets.append(
                    HOITriplet(
                        subject=hoi.subject,
                        action=action,
                        object=hoi.object,
                        confidence=conf,
                        t_start=float(hoi.t_start),
                        t_end=float(hoi.t_start + step),
                        predicted=True,
                    )
                )
        return triplets


class PredictiveHOINet(nn.Module):
    """Small MLP used to fine-tune current and future HOI prediction."""

    def __init__(self, emb_dim: int = 256, num_actions: int = 12, horizon_seconds: int = 3, hidden_dim: int = 384):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_actions = num_actions
        self.horizon_seconds = horizon_seconds
        self.encoder = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.current_head = nn.Linear(hidden_dim, num_actions)
        self.future_action_head = nn.Linear(hidden_dim, horizon_seconds * num_actions)
        self.future_emb_head = nn.Linear(hidden_dim, horizon_seconds * emb_dim)

    def forward(self, object_emb: torch.Tensor, memory_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([object_emb, memory_emb], dim=-1)
        z = self.encoder(x)
        current_logits = self.current_head(z)
        future_logits = self.future_action_head(z).reshape(-1, self.horizon_seconds, self.num_actions)
        future_emb = self.future_emb_head(z).reshape(-1, self.horizon_seconds, self.emb_dim)
        return current_logits, future_logits, future_emb


@dataclass(slots=True)
class PredictiveHOIModule:
    """ProtoHOI-style zero-shot inference with optional fine-tuned adapter."""

    future_horizon_seconds: int = 3
    emb_dim: int = 256
    subject_label: str = "human"
    memory_mix: float = 0.30
    future_step_mix: float = 0.18
    action_vocab: list[str] = field(
        default_factory=lambda: [
            "hold",
            "cut",
            "pour",
            "open",
            "touch_hot_surface",
            "carry",
            "drop",
            "close",
            "push",
            "pull",
            "move",
            "inspect",
        ]
    )
    checkpoint_path: str | None = None
    action_prototypes: torch.Tensor = field(init=False)
    _adapter: PredictiveHOINet | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.future_horizon_seconds = int(max(1, min(3, self.future_horizon_seconds)))
        self.action_prototypes = torch.stack(
            [_text_proto(action, self.emb_dim) for action in self.action_vocab], dim=0
        ).to(torch.float32)
        if self.checkpoint_path:
            self._adapter = load_predictive_hoi_checkpoint(self.checkpoint_path, device="cpu")[0]
            self._adapter.eval()

    def _memory_embedding(self, memory_state: MemoryState) -> torch.Tensor:
        return _fit_embedding(memory_state.hoi_embedding, self.emb_dim)

    def _state_hint_embedding(self, memory_state: MemoryState) -> torch.Tensor:
        state = memory_state.state_vector
        if state.numel() == 0:
            return torch.zeros((self.emb_dim,), dtype=torch.float32)
        state = state.detach().to(torch.float32).flatten()
        start = 32
        if state.shape[0] <= start:
            return torch.zeros((self.emb_dim,), dtype=torch.float32)
        out = torch.zeros((self.emb_dim,), dtype=torch.float32)
        n = min(self.emb_dim, state.shape[0] - start)
        out[:n] = state[start : start + n]
        return out

    def _detection_embedding(self, det: PerceptionDetection, fallback_seed: int) -> torch.Tensor:
        if det.clip_embedding.numel() > 0:
            return _fit_embedding(det.clip_embedding, self.emb_dim)
        # Deterministic fallback keeps output stable when explicit embeddings are unavailable.
        return _text_proto(f"{det.label}:{fallback_seed}:{det.track_id}", self.emb_dim)

    def _action_from_embedding(self, emb: torch.Tensor) -> tuple[int, float]:
        emb = _normalize(emb.to(torch.float32))
        protos = self.action_prototypes.to(emb.device)
        logits = torch.matmul(protos, emb)
        probs = torch.softmax(logits, dim=0)
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        return idx, min(1.0, max(0.0, conf))

    def _future_embeddings(
        self,
        fused: torch.Tensor,
        memory_emb: torch.Tensor,
        state_hint: torch.Tensor,
        horizon_seconds: int,
    ) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        for step in range(1, horizon_seconds + 1):
            step_mix = min(0.85, step * self.future_step_mix)
            hint_mix = min(0.25, 0.05 * step)
            pred = _normalize((1.0 - step_mix) * fused + step_mix * memory_emb + hint_mix * state_hint)
            outs.append(pred.unsqueeze(0))
        return torch.cat(outs, dim=0) if outs else torch.zeros((0, self.emb_dim), dtype=torch.float32)

    def infer(
        self,
        memory_state: MemoryState,
        object_detections: list[PerceptionDetection],
        timestamp: float,
        horizon_seconds: int | None = None,
    ) -> HOIInferenceOutput:
        horizon = int(max(1, min(3, horizon_seconds if horizon_seconds is not None else self.future_horizon_seconds)))
        if not object_detections:
            return HOIInferenceOutput(
                hoi_current=[],
                hoi_future_embeddings=torch.zeros((0, horizon, self.emb_dim), dtype=torch.float32),
                future_action_labels=[],
                future_action_confidences=torch.zeros((0, horizon), dtype=torch.float32),
            )

        memory_emb = _normalize(self._memory_embedding(memory_state))
        state_hint = _normalize(self._state_hint_embedding(memory_state))
        current: list[HOI] = []
        future_emb_rows: list[torch.Tensor] = []
        future_labels: list[list[str]] = []
        future_conf_rows: list[torch.Tensor] = []
        for i, det in enumerate(object_detections):
            det_emb = _normalize(self._detection_embedding(det, fallback_seed=i))
            fused = _normalize((1.0 - self.memory_mix) * det_emb + self.memory_mix * memory_emb)
            act_idx, act_conf = self._action_from_embedding(fused)
            current.append(
                HOI(
                    subject=self.subject_label,
                    action=self.action_vocab[act_idx],
                    object=det.label,
                    confidence=min(1.0, max(0.0, act_conf * det.confidence)),
                    t_start=float(timestamp),
                    t_end=float(timestamp),
                )
            )

            future_emb = self._future_embeddings(fused, memory_emb, state_hint, horizon)
            if self._adapter is not None:
                with torch.inference_mode():
                    dev = next(self._adapter.parameters()).device
                    cur_logits, _, learned_future = self._adapter(
                        det_emb.unsqueeze(0).to(dev), memory_emb.unsqueeze(0).to(dev)
                    )
                    learned_future = learned_future[0, :horizon].detach().cpu().to(torch.float32)
                    future_emb = _normalize(0.5 * future_emb + 0.5 * learned_future)
                    probs = torch.softmax(cur_logits[0].detach().cpu(), dim=0)
                    best = int(torch.argmax(probs).item())
                    current[-1].action = self.action_vocab[best]
                    current[-1].confidence = min(1.0, float(probs[best].item()) * det.confidence)

            labels: list[str] = []
            confs: list[float] = []
            for step_idx in range(horizon):
                fut_idx, fut_conf = self._action_from_embedding(future_emb[step_idx])
                labels.append(self.action_vocab[fut_idx])
                confs.append(fut_conf)
            future_emb_rows.append(future_emb.unsqueeze(0))
            future_labels.append(labels)
            future_conf_rows.append(torch.tensor(confs, dtype=torch.float32).unsqueeze(0))

        return HOIInferenceOutput(
            hoi_current=current,
            hoi_future_embeddings=torch.cat(future_emb_rows, dim=0),
            future_action_labels=future_labels,
            future_action_confidences=torch.cat(future_conf_rows, dim=0),
        )


def _unpack_training_batch(
    batch: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
    emb_dim: int,
    horizon_seconds: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(batch, dict):
        object_emb = batch["object_embedding"].to(device).to(torch.float32)
        memory_emb = batch.get("memory_embedding")
        if memory_emb is None:
            memory_emb = torch.zeros_like(object_emb)
        memory_emb = memory_emb.to(device).to(torch.float32)
        current_idx = batch["current_action_idx"].to(device).long()
        future_idx = batch.get("future_action_indices")
        future_emb = batch.get("future_embeddings")
        future_idx = _to_device(future_idx, device)
        future_emb = _to_device(future_emb, device)
        if future_idx is not None:
            future_idx = future_idx.long()
        if future_emb is not None:
            future_emb = future_emb.to(torch.float32)
        return object_emb, memory_emb, current_idx, future_idx, future_emb

    items = list(batch)
    if len(items) < 3:
        raise ValueError("Training batch tuple must contain at least object_emb, memory_emb, current_action_idx.")
    object_emb = items[0].to(device).to(torch.float32)
    memory_emb = items[1].to(device).to(torch.float32)
    current_idx = items[2].to(device).long()
    future_idx = items[3].to(device).long() if len(items) > 3 else None
    future_emb = items[4].to(device).to(torch.float32) if len(items) > 4 else None
    if object_emb.shape[-1] != emb_dim or memory_emb.shape[-1] != emb_dim:
        raise ValueError(f"Expected embedding dim {emb_dim}, got {object_emb.shape[-1]} and {memory_emb.shape[-1]}.")
    if future_idx is not None and future_idx.shape[-1] != horizon_seconds:
        raise ValueError(f"Expected future_action_indices width {horizon_seconds}, got {future_idx.shape[-1]}.")
    if future_emb is not None and (future_emb.shape[-2] != horizon_seconds or future_emb.shape[-1] != emb_dim):
        raise ValueError(
            f"Expected future_embeddings shape [B,{horizon_seconds},{emb_dim}], got {tuple(future_emb.shape)}."
        )
    return object_emb, memory_emb, current_idx, future_idx, future_emb


def train_predictive_hoi(
    model: PredictiveHOINet,
    train_loader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_current: float = 0.6,
    weight_future_action: float = 0.25,
    weight_future_embedding: float = 0.15,
    device: str = "cpu",
    use_amp: bool = False,
) -> dict[str, list[float]]:
    """Train/fine-tune predictive HOI network on current+future targets."""
    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {"loss": [], "current_ce": [], "future_ce": [], "future_emb": []}
    for _ in range(epochs):
        model.train()
        sum_loss = 0.0
        sum_cur = 0.0
        sum_fut = 0.0
        sum_emb = 0.0
        steps = 0
        for batch in train_loader:
            obj_emb, mem_emb, cur_idx, fut_idx, fut_emb_gt = _unpack_training_batch(
                batch, model.emb_dim, model.horizon_seconds, dev
            )
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp and dev.type != "cpu", dtype=torch.float16):
                cur_logits, fut_logits, fut_emb_pred = model(obj_emb, mem_emb)
                cur_loss = ce(cur_logits, cur_idx)
                fut_loss = torch.zeros((), dtype=torch.float32, device=dev)
                if fut_idx is not None:
                    fut_loss = ce(fut_logits.reshape(-1, model.num_actions), fut_idx.reshape(-1))
                emb_loss = torch.zeros((), dtype=torch.float32, device=dev)
                if fut_emb_gt is not None:
                    pred_n = _normalize(fut_emb_pred)
                    tgt_n = _normalize(fut_emb_gt)
                    emb_loss = 1.0 - (pred_n * tgt_n).sum(dim=-1).mean()
                loss = weight_current * cur_loss + weight_future_action * fut_loss + weight_future_embedding * emb_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            steps += 1
            sum_loss += float(loss.item())
            sum_cur += float(cur_loss.item())
            sum_fut += float(fut_loss.item())
            sum_emb += float(emb_loss.item())
        denom = max(1, steps)
        history["loss"].append(sum_loss / denom)
        history["current_ce"].append(sum_cur / denom)
        history["future_ce"].append(sum_fut / denom)
        history["future_emb"].append(sum_emb / denom)
    return history


def evaluate_predictive_hoi(
    model: PredictiveHOINet,
    eval_loader: DataLoader,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate current action accuracy and future embedding quality."""
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    cur_correct = 0
    cur_total = 0
    fut_correct = 0
    fut_total = 0
    fut_cos_sum = 0.0
    fut_cos_count = 0
    with torch.inference_mode():
        for batch in eval_loader:
            obj_emb, mem_emb, cur_idx, fut_idx, fut_emb_gt = _unpack_training_batch(
                batch, model.emb_dim, model.horizon_seconds, dev
            )
            cur_logits, fut_logits, fut_emb_pred = model(obj_emb, mem_emb)
            cur_pred = torch.argmax(cur_logits, dim=-1)
            cur_correct += int((cur_pred == cur_idx).sum().item())
            cur_total += int(cur_idx.numel())
            if fut_idx is not None:
                fut_pred = torch.argmax(fut_logits, dim=-1)
                fut_correct += int((fut_pred == fut_idx).sum().item())
                fut_total += int(fut_idx.numel())
            if fut_emb_gt is not None:
                pred_n = _normalize(fut_emb_pred)
                tgt_n = _normalize(fut_emb_gt)
                cos = (pred_n * tgt_n).sum(dim=-1)
                fut_cos_sum += float(cos.sum().item())
                fut_cos_count += int(cos.numel())
    return {
        "current_top1": float(cur_correct / cur_total) if cur_total else 0.0,
        "future_top1": float(fut_correct / fut_total) if fut_total else 0.0,
        "future_embedding_cosine": float(fut_cos_sum / fut_cos_count) if fut_cos_count else 0.0,
    }


def save_predictive_hoi_checkpoint(
    path: str | Path,
    model: PredictiveHOINet,
    *,
    action_vocab: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "emb_dim": model.emb_dim,
        "num_actions": model.num_actions,
        "horizon_seconds": model.horizon_seconds,
        "action_vocab": action_vocab,
        "extra": extra or {},
    }
    torch.save(payload, out)


def load_predictive_hoi_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[PredictiveHOINet, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    model = PredictiveHOINet(
        emb_dim=int(payload.get("emb_dim", 256)),
        num_actions=int(payload.get("num_actions", 12)),
        horizon_seconds=int(payload.get("horizon_seconds", 3)),
    )
    state_dict = payload.get("state_dict", {})
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, payload

