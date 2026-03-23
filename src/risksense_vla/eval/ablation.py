"""Ablation study framework for systematic VLA module comparison.

Defines ablation configurations, alternative module implementations (NaiveMemory,
UniformAttentionScheduler), and the AblationRunner that executes evaluation
pipelines and collects metrics across configuration variants.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from risksense_vla.attention.semantic_scheduler import SemanticAttentionScheduler
from risksense_vla.experimental import apply_occlusion
from risksense_vla.experimental import method_display_name
from risksense_vla.eval.metrics import (
    SequenceMetrics,
    aggregate_sequences,
    evaluate_sequence,
)
from risksense_vla.hoi import PredictiveHOIModule, PredictiveHOINet, ProtoHOIPredictor
from risksense_vla.memory.hazard_memory import HazardAwareMemory
from risksense_vla.train import benchmark_module, convert_qat
from risksense_vla.train.quantization import apply_int4_ptq
from risksense_vla.types import (
    HazardScore,
    MemoryObjectState,
    MemoryState,
    PerceptionDetection,
)

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Set all RNG seeds for reproducible evaluation runs."""
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VALID_MEMORY_MODES = ("hazard_aware", "naive")
_VALID_HOI_MODES = ("predictive", "frame_only")
_VALID_ATTENTION_MODES = ("semantic", "uniform")
_VALID_QUANT_MODES = ("fp32", "int8", "int4_ptq", "int8_masked")


@dataclass
class AblationConfig:
    """Specification for a single ablation experiment."""

    name: str
    memory_mode: str = "hazard_aware"
    hoi_mode: str = "predictive"
    attention_mode: str = "semantic"
    quant_mode: str = "fp32"
    seed: int = 42
    description: str = ""

    def __post_init__(self) -> None:
        if self.memory_mode not in _VALID_MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {_VALID_MEMORY_MODES}, got '{self.memory_mode}'")
        if self.hoi_mode not in _VALID_HOI_MODES:
            raise ValueError(f"hoi_mode must be one of {_VALID_HOI_MODES}, got '{self.hoi_mode}'")
        if self.attention_mode not in _VALID_ATTENTION_MODES:
            raise ValueError(f"attention_mode must be one of {_VALID_ATTENTION_MODES}, got '{self.attention_mode}'")
        if self.quant_mode not in _VALID_QUANT_MODES:
            raise ValueError(f"quant_mode must be one of {_VALID_QUANT_MODES}, got '{self.quant_mode}'")


ABLATION_REGISTRY: dict[str, AblationConfig] = {
    "baseline": AblationConfig(
        name="baseline",
        memory_mode="hazard_aware",
        hoi_mode="predictive",
        attention_mode="semantic",
        quant_mode="fp32",
        description="Full system baseline (hazard-aware SSM + predictive HOI + semantic attention + FP32)",
    ),
    "naive_memory": AblationConfig(
        name="naive_memory",
        memory_mode="naive",
        hoi_mode="predictive",
        attention_mode="semantic",
        quant_mode="fp32",
        description="Replace hazard-aware SSM memory with naive uniform-decay memory",
    ),
    "frame_only_hoi": AblationConfig(
        name="frame_only_hoi",
        memory_mode="hazard_aware",
        hoi_mode="frame_only",
        attention_mode="semantic",
        quant_mode="fp32",
        description="Replace predictive HOI with frame-by-frame ProtoHOI (no future prediction heads)",
    ),
    "uniform_attention": AblationConfig(
        name="uniform_attention",
        memory_mode="hazard_aware",
        hoi_mode="predictive",
        attention_mode="uniform",
        quant_mode="fp32",
        description="Replace semantic attention scheduling with uniform allocation",
    ),
    "int8_qat": AblationConfig(
        name="int8_qat",
        memory_mode="hazard_aware",
        hoi_mode="predictive",
        attention_mode="semantic",
        quant_mode="int8",
        description="INT8 quantization via QAT convert",
    ),
    "int4_ptq": AblationConfig(
        name="int4_ptq",
        memory_mode="hazard_aware",
        hoi_mode="predictive",
        attention_mode="semantic",
        quant_mode="int4_ptq",
        description="INT4-style post-training quantization with dynamic quantization",
    ),
    "int8_masked": AblationConfig(
        name="int8_masked",
        memory_mode="hazard_aware",
        hoi_mode="predictive",
        attention_mode="semantic",
        quant_mode="int8_masked",
        description="INT8 QAT combined with weight pruning/masking",
    ),
}


def load_ablation_configs_from_yaml(path: str | Path) -> dict[str, AblationConfig]:
    """Load ablation config overrides from a YAML file.

    Non-specified fields inherit from the baseline defaults.
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    global_seed = int(data.get("seed", 42))
    baseline_defaults = {
        "memory_mode": "hazard_aware",
        "hoi_mode": "predictive",
        "attention_mode": "semantic",
        "quant_mode": "fp32",
    }
    configs: dict[str, AblationConfig] = {}
    for name, overrides in data.get("ablations", {}).items():
        merged = {**baseline_defaults, **(overrides or {})}
        configs[name] = AblationConfig(
            name=name,
            seed=int(merged.pop("seed", global_seed)),
            description=str(merged.pop("description", "")),
            **merged,
        )
    return configs


# ---------------------------------------------------------------------------
# Alternative module implementations for ablation variants
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _copy_object_state(obj: MemoryObjectState) -> MemoryObjectState:
    return MemoryObjectState(
        track_id=obj.track_id,
        label=obj.label,
        last_bbox_xyxy=tuple(obj.last_bbox_xyxy),
        persistence=float(obj.persistence),
        hazard_weight=float(obj.hazard_weight),
        age_frames=int(obj.age_frames),
    )


class NaiveMemory:
    """Uniform-decay memory without hazard-aware SSM gating.

    Drop-in replacement for HazardAwareMemory that uses constant decay
    and equal weighting for all detections, regardless of hazard signals.
    This isolates the contribution of hazard-conditioned memory dynamics.
    """

    def __init__(self, emb_dim: int = 256, base_decay: float = 0.86) -> None:
        self.emb_dim = emb_dim
        self.base_decay = base_decay
        self.min_persistence = 0.05
        self.max_persistence = 1.0
        self.observation_boost = 0.20
        self.objects: dict[str, MemoryObjectState] = {}
        self.hoi_embedding = torch.zeros((1, emb_dim), dtype=torch.float32)

    def update(
        self,
        timestamp: float,
        detections: list[PerceptionDetection],
        hazards: list[float] | None = None,
        hazard_events: list[HazardScore] | None = None,
        previous_memory_state: MemoryState | None = None,
        log_callback: Callable[[dict[str, float]], None] | None = None,
    ) -> MemoryState:
        _ = hazards, hazard_events, log_callback
        if previous_memory_state is not None:
            self._load_previous(previous_memory_state)

        seen: set[str] = set()
        for det in detections:
            seen.add(det.track_id)
            obj = self.objects.get(det.track_id)
            if obj is None:
                obj = MemoryObjectState(
                    track_id=det.track_id,
                    label=det.label,
                    last_bbox_xyxy=det.bbox_xyxy,
                    persistence=0.65,
                    hazard_weight=0.0,
                    age_frames=1,
                )
            else:
                obj.last_bbox_xyxy = det.bbox_xyxy
                obj.age_frames += 1
                obj.persistence = _clamp(
                    obj.persistence * self.base_decay + self.observation_boost,
                    self.min_persistence,
                    self.max_persistence,
                )
            self.objects[det.track_id] = obj

        for track_id in tuple(self.objects):
            if track_id in seen:
                continue
            obj = self.objects[track_id]
            obj.persistence = _clamp(obj.persistence * self.base_decay, 0.0, self.max_persistence)
            if obj.persistence < self.min_persistence:
                del self.objects[track_id]
            else:
                self.objects[track_id] = obj

        # Simple mean-pool embedding update (no SSM, no hazard gating)
        if detections:
            emb_sum = torch.zeros((1, self.emb_dim), dtype=torch.float32)
            for det in detections:
                e = torch.zeros((1, self.emb_dim), dtype=torch.float32)
                if det.clip_embedding.numel() > 0:
                    raw = det.clip_embedding.to(torch.float32).flatten()
                    n = min(self.emb_dim, raw.shape[0])
                    e[0, :n] = raw[:n]
                emb_sum += e
            avg_emb = emb_sum / max(1, len(detections))
            self.hoi_embedding = 0.8 * self.hoi_embedding + 0.2 * avg_emb

        state_vector = torch.zeros((1, 512), dtype=torch.float32)
        objs = list(self.objects.values())
        if objs:
            state_vector[0, 0] = float(len(objs))
            state_vector[0, 1] = float(sum(o.persistence for o in objs) / len(objs))
        copy_n = min(self.emb_dim, 512 - 32)
        state_vector[0, 32 : 32 + copy_n] = self.hoi_embedding[0, :copy_n]

        return MemoryState(
            timestamp=timestamp,
            objects=[_copy_object_state(o) for o in sorted(self.objects.values(), key=lambda x: x.track_id)],
            hoi_embedding=self.hoi_embedding.clone(),
            state_vector=state_vector,
        )

    def _load_previous(self, ms: MemoryState) -> None:
        self.objects = {o.track_id: _copy_object_state(o) for o in ms.objects}
        if ms.hoi_embedding.numel() > 0:
            e = ms.hoi_embedding.to(torch.float32)
            if e.ndim == 1:
                e = e.unsqueeze(0)
            out = torch.zeros((1, self.emb_dim), dtype=torch.float32)
            n = min(self.emb_dim, e.shape[-1])
            out[0, :n] = e[0, :n]
            self.hoi_embedding = out


class UniformAttentionScheduler:
    """Attention scheduler that assigns equal compute to all detections.

    Drop-in replacement for SemanticAttentionScheduler; returns 1.0 for
    every detection regardless of hazard risk.
    """

    def allocation(
        self, detections: list[PerceptionDetection], hazards: list[HazardScore]
    ) -> dict[str, float]:
        return {det.track_id: 1.0 for det in detections}


# ---------------------------------------------------------------------------
# Ablation result container
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Metrics collected for a single ablation configuration."""

    config_name: str
    config: AblationConfig
    thc: float = 0.0
    haa: float = 0.0
    rme: float = 0.0
    detection_map: float = 0.0
    fps: float = 0.0
    latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    prediction_accuracy_1s: float = 0.0
    prediction_accuracy_2s: float = 0.0
    prediction_accuracy_3s: float = 0.0
    hazard_lead_time_mean: float = 0.0
    hazard_lead_time_median: float = 0.0
    per_module_fps: dict[str, float] = field(default_factory=dict)
    per_module_latency_ms: dict[str, float] = field(default_factory=dict)
    seed: int = 42

    def as_flat_dict(self) -> dict[str, Any]:
        return {
            "ablation": self.config_name,
            "method_name": method_display_name(
                "naive"
                if self.config.memory_mode == "naive"
                else "frame_only"
                if self.config.hoi_mode == "frame_only"
                else "hazard_aware"
            ),
            "memory_mode": self.config.memory_mode,
            "hoi_mode": self.config.hoi_mode,
            "attention_mode": self.config.attention_mode,
            "quant_mode": self.config.quant_mode,
            "seed": self.seed,
            "THC": round(self.thc, 4),
            "HAA": round(self.haa, 4),
            "RME": round(self.rme, 4),
            "mAP": round(self.detection_map, 4),
            "FPS": round(self.fps, 2),
            "latency_ms": round(self.latency_ms, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "prediction_accuracy@1s": round(self.prediction_accuracy_1s, 4),
            "prediction_accuracy@2s": round(self.prediction_accuracy_2s, 4),
            "prediction_accuracy@3s": round(self.prediction_accuracy_3s, 4),
            "HazardLeadTimeMean": round(self.hazard_lead_time_mean, 4),
            "HazardLeadTimeMedian": round(self.hazard_lead_time_median, 4),
        }


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def _build_memory(config: AblationConfig, cfg: dict[str, Any]) -> HazardAwareMemory | NaiveMemory:
    emb_dim = int(cfg.get("perception", {}).get("embedding_dim", 256))
    if config.memory_mode == "naive":
        return NaiveMemory(emb_dim=emb_dim)
    use_hazard_weighting = bool(cfg.get("memory", {}).get("use_hazard_weighting", True))
    return HazardAwareMemory(
        emb_dim=emb_dim,
        alpha=0.0 if not use_hazard_weighting else 0.14,
        beta=0.0 if not use_hazard_weighting else 1.4,
        use_hazard_weighting=use_hazard_weighting,
    )


def _build_hoi(
    config: AblationConfig, cfg: dict[str, Any]
) -> PredictiveHOIModule | ProtoHOIPredictor:
    emb_dim = int(cfg.get("perception", {}).get("embedding_dim", 256))
    horizon = int(cfg.get("hazard", {}).get("future_horizon_seconds", 3))
    if config.hoi_mode == "frame_only":
        return ProtoHOIPredictor(future_horizon_seconds=horizon, emb_dim=emb_dim)
    return PredictiveHOIModule(future_horizon_seconds=horizon, emb_dim=emb_dim)


def _build_attention(
    config: AblationConfig, cfg: dict[str, Any]
) -> SemanticAttentionScheduler | UniformAttentionScheduler:
    if config.attention_mode == "uniform":
        return UniformAttentionScheduler()
    att_cfg = cfg.get("attention", {})
    return SemanticAttentionScheduler(
        threshold=float(att_cfg.get("semantic_attention_threshold", 0.6)),
        low_risk_scale=float(att_cfg.get("low_risk_scale", 0.5)),
        high_risk_scale=float(att_cfg.get("high_risk_scale", 1.0)),
    )


def _apply_quantization(
    model: nn.Module,
    config: AblationConfig,
    cfg: dict[str, Any],
) -> nn.Module:
    """Apply quantization to a model based on the ablation config."""
    if config.quant_mode == "fp32":
        return model

    if config.quant_mode in ("int8", "int8_masked"):
        if config.quant_mode == "int8_masked":
            pruning_ratio = float(cfg.get("optimization", {}).get("pruning_ratio", 0.2))
            _apply_magnitude_pruning(model, pruning_ratio)
        return convert_qat(model)

    if config.quant_mode == "int4_ptq":
        emb_dim = int(cfg.get("perception", {}).get("embedding_dim", 256))
        dummy_loader = _make_calibration_loader(emb_dim, batches=8)
        return apply_int4_ptq(model, dummy_loader, {"int4": {"enabled": True, "calibration_batches": 8}})

    return model


def _apply_magnitude_pruning(model: nn.Module, ratio: float) -> None:
    """Zero out the smallest weights by magnitude (unstructured pruning)."""
    if ratio <= 0.0:
        return
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        with torch.no_grad():
            flat = param.abs().flatten()
            k = int(flat.numel() * ratio)
            if k == 0:
                continue
            threshold = torch.kthvalue(flat, k).values
            mask = (param.abs() >= threshold).float()
            param.mul_(mask)


def _make_calibration_loader(
    emb_dim: int, batches: int = 8, batch_size: int = 16
) -> torch.utils.data.DataLoader:
    """Create a synthetic calibration DataLoader for INT4 PTQ.

    Produces (object_emb, memory_emb, label) tuples matching
    ``PredictiveHOINet.forward`` signature so that ``_run_calibration``
    correctly unpacks the batch.
    """
    from torch.utils.data import DataLoader, TensorDataset

    n = batches * batch_size
    obj = torch.randn(n, emb_dim)
    mem = torch.randn(n, emb_dim)
    labels = torch.zeros(n, dtype=torch.long)
    ds = TensorDataset(obj, mem, labels)
    return DataLoader(ds, batch_size=batch_size)


@dataclass
class AblationPipeline:
    """Assembled pipeline components for a single ablation variant."""

    config: AblationConfig
    memory: HazardAwareMemory | NaiveMemory
    hoi: PredictiveHOIModule | ProtoHOIPredictor
    attention: SemanticAttentionScheduler | UniformAttentionScheduler


def build_pipeline(config: AblationConfig, cfg: dict[str, Any]) -> AblationPipeline:
    """Construct the evaluation pipeline for the given ablation config."""
    return AblationPipeline(
        config=config,
        memory=_build_memory(config, cfg),
        hoi=_build_hoi(config, cfg),
        attention=_build_attention(config, cfg),
    )


# ---------------------------------------------------------------------------
# Synthetic benchmark data generation
# ---------------------------------------------------------------------------


def _generate_synthetic_sequence(
    num_frames: int = 100,
    emb_dim: int = 256,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a synthetic frame-log sequence for metrics evaluation.

    Produces JSONL-compatible records with detections, HOIs, hazards,
    attention allocation, and latency fields.
    """
    rng = np.random.RandomState(seed)
    records: list[dict[str, Any]] = []
    actions = ["hold", "cut", "pour", "open", "touch_hot_surface", "carry", "drop"]

    for frame_id in range(num_frames):
        n_det = rng.randint(1, 6)
        detections = []
        for d in range(n_det):
            detections.append({
                "track_id": f"obj_{d}",
                "label": rng.choice(["knife", "glass", "bottle", "stove", "person"]),
                "confidence": float(rng.uniform(0.5, 1.0)),
                "bbox_xyxy": [int(x) for x in rng.randint(0, 500, 4).tolist()],
            })

        n_hoi = rng.randint(0, 4)
        hois = []
        for _ in range(n_hoi):
            is_predicted = bool(rng.random() < 0.3)
            hois.append({
                "subject": "human",
                "action": rng.choice(actions),
                "object": rng.choice(["knife", "glass", "bottle"]),
                "confidence": float(rng.uniform(0.3, 1.0)),
                "predicted": is_predicted,
            })

        hazard_score = float(rng.uniform(0.0, 1.0))
        hazards = []
        if hazard_score > 0.5:
            hazards.append({
                "subject": "human",
                "action": rng.choice(actions),
                "object": rng.choice(["knife", "stove"]),
                "score": hazard_score,
            })

        att_alloc = {f"obj_{d}": float(rng.uniform(0.4, 1.0)) for d in range(n_det)}

        lat = {
            "perception": float(rng.uniform(10.0, 60.0)),
            "memory": float(rng.uniform(1.0, 8.0)),
            "hoi": float(rng.uniform(5.0, 25.0)),
        }

        records.append({
            "frame_id": frame_id,
            "detections": detections,
            "hois": hois,
            "hazards": hazards,
            "attention_allocation": att_alloc,
            "latency_ms": lat,
        })

    return records


def _load_benchmark_sequences(
    dataset_dir: str | Path,
    emb_dim: int = 256,
    seed: int = 42,
) -> list[list[dict[str, Any]]]:
    """Load benchmark sequences from a dataset directory or generate synthetic ones.

    Looks for ``*.jsonl`` files in *dataset_dir*. Each file is treated as
    one sequence. Falls back to synthetic generation when no files are found.
    """
    dataset_path = Path(dataset_dir)
    sequences: list[list[dict[str, Any]]] = []

    if dataset_path.is_dir():
        jsonl_files = sorted(dataset_path.glob("*.jsonl"))
        for jf in jsonl_files:
            recs = []
            for line in jf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
            if recs:
                sequences.append(recs)

    if not sequences:
        _LOG.info("No JSONL sequences found in %s; generating synthetic data", dataset_dir)
        sequences.append(_generate_synthetic_sequence(num_frames=200, emb_dim=emb_dim, seed=seed))
        sequences.append(_generate_synthetic_sequence(num_frames=150, emb_dim=emb_dim, seed=seed + 1))

    return sequences


# ---------------------------------------------------------------------------
# Pipeline execution helpers
# ---------------------------------------------------------------------------


def _run_pipeline_on_sequence(
    pipeline: AblationPipeline,
    records: list[dict[str, Any]],
    emb_dim: int = 256,
    *,
    occlusion_prob: float = 0.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Execute the ablation pipeline over a sequence of frame records.

    Runs memory update, HOI inference, and attention allocation on each
    frame, then injects the resulting latency/metric signals back into the
    record for downstream metric computation.
    """
    memory = pipeline.memory
    memory_state: MemoryState | None = None
    processed: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for rec in records:
        frame_id = int(rec.get("frame_id", 0))
        timestamp = float(frame_id) / 24.0

        raw_dets = rec.get("detections", [])
        detections: list[PerceptionDetection] = []
        for d in raw_dets:
            bbox = tuple(d.get("bbox_xyxy", [0, 0, 10, 10]))
            det = PerceptionDetection(
                track_id=str(d.get("track_id", "obj_0")),
                label=str(d.get("label", "object")),
                confidence=float(d.get("confidence", 0.5)),
                bbox_xyxy=bbox,
                mask=torch.zeros((1, 1), dtype=torch.float32),
                clip_embedding=torch.randn(emb_dim, dtype=torch.float32),
            )
            detections.append(det)
        detections, occlusion_events = apply_occlusion(detections, occlusion_prob=occlusion_prob, rng=rng)

        raw_hazards = rec.get("hazards", [])
        hazard_scores = [
            HazardScore(
                subject=str(h.get("subject", "")),
                action=str(h.get("action", "")),
                object=str(h.get("object", "")),
                score=float(h.get("score", 0.0)),
                severity="medium",
                explanation="",
            )
            for h in raw_hazards
        ]
        hazard_floats = [float(h.get("score", 0.0)) for h in raw_hazards]

        t0 = time.perf_counter()
        memory_state = memory.update(
            timestamp=timestamp,
            detections=detections,
            hazards=hazard_floats if hazard_floats else None,
            hazard_events=hazard_scores,
            previous_memory_state=memory_state,
        )
        mem_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if isinstance(pipeline.hoi, PredictiveHOIModule):
            hoi_out = pipeline.hoi.infer(memory_state, detections, timestamp)
            hoi_triplets = hoi_out.as_triplets()
        else:
            hoi_triplets = pipeline.hoi.predict(timestamp, detections, memory_state)
        hoi_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        att_alloc = pipeline.attention.allocation(detections, hazard_scores)
        att_ms = (time.perf_counter() - t0) * 1000.0

        out_hois = []
        for t in hoi_triplets:
            out_hois.append({
                "subject": t.subject,
                "action": t.action,
                "object": t.object,
                "confidence": float(t.confidence),
                "predicted": bool(t.predicted),
                "t_start": float(t.t_start),
                "t_end": float(t.t_end),
            })

        processed.append({
            "frame_id": frame_id,
            "detections": raw_dets,
            "hois": out_hois,
            "hazards": raw_hazards,
            "attention_allocation": {k: float(v) for k, v in att_alloc.items()},
            "latency_ms": {
                "perception": float(rec.get("latency_ms", {}).get("perception", 20.0)),
                "memory": mem_ms,
                "hoi": hoi_ms,
                "attention": att_ms,
            },
            "occlusion_events": occlusion_events,
        })

    return processed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AblationRunner:
    """Orchestrates ablation experiments across multiple configurations."""

    def __init__(
        self,
        cfg: dict[str, Any],
        dataset_dir: str | Path = "outputs",
        warmup: int = 50,
        iterations: int = 200,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.dataset_dir = Path(dataset_dir)
        self.warmup = warmup
        self.iterations = iterations
        self.device = device
        self.emb_dim = int(cfg.get("perception", {}).get("embedding_dim", 256))

    def run_single(self, config: AblationConfig) -> AblationResult:
        """Run a single ablation configuration and return metrics."""
        _LOG.info("Running ablation: %s", config.name)
        seed_everything(config.seed)

        pipeline = build_pipeline(config, self.cfg)
        sequences = _load_benchmark_sequences(
            self.dataset_dir, emb_dim=self.emb_dim, seed=config.seed
        )

        seq_metrics: list[SequenceMetrics] = []
        for seq_records in sequences:
            processed = _run_pipeline_on_sequence(
                pipeline,
                seq_records,
                self.emb_dim,
                occlusion_prob=float(self.cfg.get("evaluation", {}).get("occlusion_prob", 0.0)),
                seed=config.seed,
            )
            sm = evaluate_sequence(processed)
            seq_metrics.append(sm)

        agg = aggregate_sequences(seq_metrics)

        per_module_fps: dict[str, float] = {}
        per_module_latency: dict[str, float] = {}
        peak_mem = 0.0

        hoi_model = self._get_hoi_model(config)
        if hoi_model is not None:
            hoi_model = _apply_quantization(hoi_model, config, self.cfg)
            dummy = (torch.randn(1, self.emb_dim), torch.randn(1, self.emb_dim))
            bm = benchmark_module(
                hoi_model, dummy,
                warmup=self.warmup,
                iterations=self.iterations,
                device=self.device,
            )
            per_module_fps["hoi"] = bm["fps"]
            per_module_latency["hoi"] = bm["avg_latency_ms"]
            peak_mem = max(peak_mem, bm["peak_memory_mb"])

        return AblationResult(
            config_name=config.name,
            config=config,
            thc=agg["THC"],
            haa=agg["HAA"],
            rme=agg["RME"],
            detection_map=agg["mAP"],
            fps=agg["FPS"],
            latency_ms=agg["LatencyMS"],
            peak_memory_mb=peak_mem,
            prediction_accuracy_1s=agg.get("prediction_accuracy@1s", 0.0),
            prediction_accuracy_2s=agg.get("prediction_accuracy@2s", 0.0),
            prediction_accuracy_3s=agg.get("prediction_accuracy@3s", 0.0),
            hazard_lead_time_mean=agg.get("HazardLeadTimeMean", 0.0),
            hazard_lead_time_median=agg.get("HazardLeadTimeMedian", 0.0),
            per_module_fps=per_module_fps,
            per_module_latency_ms=per_module_latency,
            seed=config.seed,
        )

    def _get_hoi_model(self, config: AblationConfig) -> nn.Module | None:
        if config.hoi_mode == "frame_only":
            return None
        return PredictiveHOINet(emb_dim=self.emb_dim)

    def run_all(self, configs: list[AblationConfig]) -> list[AblationResult]:
        """Run all provided ablation configurations sequentially."""
        results: list[AblationResult] = []
        for config in configs:
            result = self.run_single(config)
            results.append(result)
            _LOG.info(
                "  %s: THC=%.3f HAA=%.3f RME=%.3f FPS=%.1f",
                config.name, result.thc, result.haa, result.rme, result.fps,
            )
        return results

    def run_multi_seed(
        self,
        config: AblationConfig,
        seeds: list[int],
    ) -> MultiSeedResult:
        """Run the same config across multiple seeds, returning aggregated stats."""
        per_seed: list[AblationResult] = []
        for seed in seeds:
            cfg_copy = AblationConfig(
                name=config.name,
                memory_mode=config.memory_mode,
                hoi_mode=config.hoi_mode,
                attention_mode=config.attention_mode,
                quant_mode=config.quant_mode,
                seed=seed,
                description=config.description,
            )
            per_seed.append(self.run_single(cfg_copy))
        return MultiSeedResult.from_results(config.name, per_seed)

    def run_all_multi_seed(
        self,
        configs: list[AblationConfig],
        seeds: list[int],
    ) -> list[MultiSeedResult]:
        """Run all configs across multiple seeds."""
        results: list[MultiSeedResult] = []
        for config in configs:
            ms = self.run_multi_seed(config, seeds)
            _LOG.info(
                "  %s: THC=%.3f±%.3f HAA=%.3f±%.3f RME=%.3f±%.3f FPS=%.1f±%.1f",
                config.name,
                ms.mean_thc, ms.std_thc,
                ms.mean_haa, ms.std_haa,
                ms.mean_rme, ms.std_rme,
                ms.mean_fps, ms.std_fps,
            )
            results.append(ms)
        return results


# ---------------------------------------------------------------------------
# Multi-seed aggregation and statistical testing
# ---------------------------------------------------------------------------


@dataclass
class MultiSeedResult:
    """Aggregated metrics across multiple seeds for one ablation config."""

    config_name: str
    seeds: list[int]
    per_seed_results: list[AblationResult]
    mean_thc: float = 0.0
    std_thc: float = 0.0
    mean_haa: float = 0.0
    std_haa: float = 0.0
    mean_rme: float = 0.0
    std_rme: float = 0.0
    mean_fps: float = 0.0
    std_fps: float = 0.0
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    mean_map: float = 0.0
    std_map: float = 0.0
    ci95_thc: tuple[float, float] = (0.0, 0.0)
    ci95_haa: tuple[float, float] = (0.0, 0.0)
    ci95_rme: tuple[float, float] = (0.0, 0.0)
    ci95_fps: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_results(cls, name: str, results: list[AblationResult]) -> MultiSeedResult:
        """Compute mean, std, and 95% CI from per-seed results."""
        from scipy import stats as sp_stats

        seeds = [r.seed for r in results]
        n = len(results)

        def _stats(vals: list[float]) -> tuple[float, float, tuple[float, float]]:
            arr = np.array(vals, dtype=np.float64)
            m = float(arr.mean())
            s = float(arr.std(ddof=1)) if n > 1 else 0.0
            if n > 1:
                t_val = float(sp_stats.t.ppf(0.975, df=n - 1))
                margin = t_val * s / n**0.5
                ci = (m - margin, m + margin)
            else:
                ci = (m, m)
            return m, s, ci

        thc_m, thc_s, thc_ci = _stats([r.thc for r in results])
        haa_m, haa_s, haa_ci = _stats([r.haa for r in results])
        rme_m, rme_s, rme_ci = _stats([r.rme for r in results])
        fps_m, fps_s, fps_ci = _stats([r.fps for r in results])
        lat_m, lat_s, _ = _stats([r.latency_ms for r in results])
        map_m, map_s, _ = _stats([r.detection_map for r in results])

        return cls(
            config_name=name,
            seeds=seeds,
            per_seed_results=results,
            mean_thc=thc_m, std_thc=thc_s,
            mean_haa=haa_m, std_haa=haa_s,
            mean_rme=rme_m, std_rme=rme_s,
            mean_fps=fps_m, std_fps=fps_s,
            mean_latency_ms=lat_m, std_latency_ms=lat_s,
            mean_map=map_m, std_map=map_s,
            ci95_thc=thc_ci, ci95_haa=haa_ci,
            ci95_rme=rme_ci, ci95_fps=fps_ci,
        )

    def as_flat_dict(self) -> dict[str, Any]:
        return {
            "ablation": self.config_name,
            "seeds": self.seeds,
            "n_seeds": len(self.seeds),
            "THC_mean": round(self.mean_thc, 4),
            "THC_std": round(self.std_thc, 4),
            "THC_ci95_lo": round(self.ci95_thc[0], 4),
            "THC_ci95_hi": round(self.ci95_thc[1], 4),
            "HAA_mean": round(self.mean_haa, 4),
            "HAA_std": round(self.std_haa, 4),
            "HAA_ci95_lo": round(self.ci95_haa[0], 4),
            "HAA_ci95_hi": round(self.ci95_haa[1], 4),
            "RME_mean": round(self.mean_rme, 4),
            "RME_std": round(self.std_rme, 4),
            "FPS_mean": round(self.mean_fps, 2),
            "FPS_std": round(self.std_fps, 2),
        }


def compute_significance(
    baseline: MultiSeedResult,
    variant: MultiSeedResult,
    metric: str = "thc",
) -> dict[str, float]:
    """Compute paired t-test p-value and Cohen's d effect size."""
    from scipy import stats as sp_stats

    base_vals = np.array([getattr(r, metric) for r in baseline.per_seed_results])
    var_vals = np.array([getattr(r, metric) for r in variant.per_seed_results])

    n = min(len(base_vals), len(var_vals))
    base_vals = base_vals[:n]
    var_vals = var_vals[:n]

    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0}

    t_stat, p_value = sp_stats.ttest_rel(base_vals, var_vals)
    diff = var_vals - base_vals
    cohens_d = float(diff.mean() / max(diff.std(ddof=1), 1e-12))

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
    }


def multi_seed_results_to_csv(
    results: list[MultiSeedResult], path: str | Path,
) -> None:
    """Write multi-seed ablation results to CSV."""
    cols = [
        "ablation", "n_seeds", "THC_mean", "THC_std",
        "THC_ci95_lo", "THC_ci95_hi",
        "HAA_mean", "HAA_std", "RME_mean", "RME_std",
        "FPS_mean", "FPS_std",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            row = r.as_flat_dict()
            writer.writerow({k: row.get(k, "") for k in cols})


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------


_CSV_COLUMNS = [
    "ablation",
    "method_name",
    "memory_mode",
    "hoi_mode",
    "attention_mode",
    "quant_mode",
    "seed",
    "THC",
    "HAA",
    "RME",
    "mAP",
    "FPS",
    "latency_ms",
    "peak_memory_mb",
    "prediction_accuracy@1s",
    "prediction_accuracy@2s",
    "prediction_accuracy@3s",
    "HazardLeadTimeMean",
    "HazardLeadTimeMedian",
    "delta_THC_pct",
    "delta_HAA_pct",
    "delta_RME_pct",
    "delta_FPS_pct",
]


def _compute_deltas(
    results: list[AblationResult],
) -> list[dict[str, Any]]:
    """Compute %-change relative to baseline for each result."""
    baseline: AblationResult | None = None
    for r in results:
        if r.config_name == "baseline":
            baseline = r
            break

    rows: list[dict[str, Any]] = []
    for r in results:
        row = r.as_flat_dict()
        if baseline is not None and r.config_name != "baseline":
            for metric in ("THC", "HAA", "RME", "FPS"):
                base_val = getattr(baseline, metric.lower() if metric != "mAP" else "detection_map")
                cur_val = row[metric]
                if base_val != 0:
                    row[f"delta_{metric}_pct"] = round(((cur_val - base_val) / abs(base_val)) * 100.0, 2)
                else:
                    row[f"delta_{metric}_pct"] = 0.0
        else:
            for metric in ("THC", "HAA", "RME", "FPS"):
                row[f"delta_{metric}_pct"] = 0.0
        rows.append(row)
    return rows


def results_to_csv(results: list[AblationResult], path: str | Path) -> None:
    """Write ablation results to CSV with delta columns."""
    rows = _compute_deltas(results)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            filtered = {k: row.get(k, "") for k in _CSV_COLUMNS}
            writer.writerow(filtered)


def results_to_dataframe(results: list[AblationResult]) -> Any:
    """Convert results to a pandas DataFrame (optional dependency)."""
    try:
        import pandas as pd
        rows = _compute_deltas(results)
        return pd.DataFrame(rows, columns=_CSV_COLUMNS)
    except ImportError:
        return _compute_deltas(results)


def print_summary_table(results: list[AblationResult]) -> str:
    """Format results as a human-readable ASCII table."""
    rows = _compute_deltas(results)
    header = f"{'Ablation':>20s}  {'THC':>7s}  {'HAA':>7s}  {'RME':>7s}  {'mAP':>7s}  {'FPS':>8s}  {'Lat(ms)':>8s}  {'dTHC%':>7s}  {'dHAA%':>7s}  {'dFPS%':>7s}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['ablation']:>20s}  {r['THC']:7.4f}  {r['HAA']:7.4f}  {r['RME']:7.4f}  "
            f"{r['mAP']:7.4f}  {r['FPS']:8.2f}  {r['latency_ms']:8.2f}  "
            f"{r.get('delta_THC_pct', 0):7.2f}  {r.get('delta_HAA_pct', 0):7.2f}  "
            f"{r.get('delta_FPS_pct', 0):7.2f}"
        )
    table = "\n".join(lines)
    _LOG.info("\n%s", table)
    return table
