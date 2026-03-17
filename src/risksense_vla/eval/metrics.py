"""Evaluation metrics and diagnostic visualizations for RiskSense-VLA."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameMetrics:
    """Per-frame metrics: detection count, HOI count, max hazard score, and average latency."""

    frame_id: int
    num_detections: int
    num_hois: int
    max_hazard: float
    avg_latency_ms: float


@dataclass(slots=True)
class SequenceMetrics:
    """Sequence-level metrics: THC, HAA, RME, FPS, latency, and detection mAP."""

    thc: float
    haa: float
    rme: float
    fps: float
    latency_ms: float
    detection_map: float


def _safe_mean(values: list[float]) -> float:
    """Return the mean of values, or 0.0 if the list is empty."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def frame_metrics(record: dict[str, Any]) -> FrameMetrics:
    """Extract per-frame metrics (detections, HOIs, max hazard, latency) from a record dict."""
    hazards = record.get("hazards", [])
    lat = record.get("latency_ms", {})
    return FrameMetrics(
        frame_id=int(record.get("frame_id", -1)),
        num_detections=len(record.get("detections", [])),
        num_hois=len(record.get("hois", [])),
        max_hazard=max([float(h.get("score", 0.0)) for h in hazards], default=0.0),
        avg_latency_ms=_safe_mean([float(v) for v in lat.values()]) if lat else 0.0,
    )


def temporal_hoi_consistency(records: list[dict[str, Any]]) -> float:
    r"""Temporal HOI Consistency (THC).

    Measures the proportion of consecutive frames where the top (highest-
    confidence) *observed* HOI action remains unchanged.

    .. math::

        \text{THC} = \frac{1}{T-1} \sum_{t=2}^{T}
        \mathbf{1}[a_t^{*} = a_{t-1}^{*}]

    where :math:`a_t^{*}` is the top observed action at frame *t*.  Frames
    without any observed HOI are skipped.  Returns 0.0 when fewer than two
    frames contain observed HOIs.
    """
    if len(records) < 2:
        return 0.0
    same = 0
    total = 0
    prev = None
    for rec in records:
        hois = [h for h in rec.get("hois", []) if not h.get("predicted", False)]
        if not hois:
            continue
        cur = max(hois, key=lambda h: float(h.get("confidence", 0.0))).get("action", "")
        if prev is not None:
            total += 1
            if cur == prev:
                same += 1
        prev = cur
    return float(same / total) if total else 0.0


def hazard_anticipation_accuracy(
    records: list[dict[str, Any]],
    lead_frames: int = 25,
    hazard_threshold: float = 0.7,
) -> float:
    r"""Hazard Anticipation Accuracy (HAA).

    Measures how often a predicted HOI appears within ``lead_frames``
    frames *before* a genuine hazardous event.

    .. math::

        \text{HAA} = \frac{1}{|H|} \sum_{h \in H}
        \mathbf{1}\bigl[\exists\, p \in P :
        h - L \le p \le h\bigr]

    where :math:`H` is the set of frames with hazard score
    :math:`\ge` ``hazard_threshold``, :math:`P` is the set of frames
    containing a predicted HOI, and :math:`L` = ``lead_frames``.
    Returns 0.0 when no hazardous frames exist.
    """
    hazard_frames = []
    pred_frames = []
    for rec in records:
        frame_id = int(rec.get("frame_id", -1))
        for h in rec.get("hazards", []):
            if float(h.get("score", 0.0)) >= hazard_threshold:
                if h.get("action", ""):
                    hazard_frames.append(frame_id)
        for hoi in rec.get("hois", []):
            if hoi.get("predicted", False):
                pred_frames.append(frame_id)
    if not hazard_frames:
        return 0.0
    hits = 0
    for hf in hazard_frames:
        if any((hf - lead_frames) <= pf <= hf for pf in pred_frames):
            hits += 1
    return float(hits / len(hazard_frames))


def risk_weighted_memory_efficiency(records: list[dict[str, Any]]) -> float:
    r"""Risk-weighted Memory Efficiency (RME).

    Measures alignment between compute allocation and hazard intensity.
    Higher values indicate that more compute is spent on genuinely
    hazardous frames.

    .. math::

        \text{RME} = \operatorname{clip}\!\Bigl(
        \frac{1}{T} \sum_{t=1}^{T}
        \bar{s}_t \cdot \bar{c}_t,\; 0,\; 1\Bigr)

    where :math:`\bar{s}_t` is the mean hazard score at frame *t* and
    :math:`\bar{c}_t` is the mean attention compute allocation.
    Clipped to [0, 1] because both factors are individually bounded.
    """
    alloc = []
    for rec in records:
        att = rec.get("attention_allocation", {})
        hz = rec.get("hazards", [])
        score = _safe_mean([float(h.get("score", 0.0)) for h in hz])
        avg_compute = _safe_mean([float(v) for v in att.values()]) if att else 0.0
        alloc.append(score * avg_compute)
    if not alloc:
        return 0.0
    return float(np.clip(np.mean(alloc), 0.0, 1.0))


def detection_map_stub(records: list[dict[str, Any]]) -> float:
    """Placeholder mAP: mean detection confidence (NOT a true mAP).

    This stub returns the average detection confidence across all frames
    as a rough proxy.  Replace with a proper mAP computation once a
    ground-truth benchmark adapter is connected.
    """
    _LOG.debug("detection_map_stub called; this is NOT a true mAP computation")
    confs = []
    for rec in records:
        confs.extend(float(d.get("confidence", 0.0)) for d in rec.get("detections", []))
    return _safe_mean(confs)


def evaluate_sequence(
    records: list[dict[str, Any]],
    *,
    hazard_threshold: float = 0.7,
    lead_frames: int = 25,
) -> SequenceMetrics:
    """Compute all sequence-level metrics from a list of frame records."""
    lat = [frame_metrics(r).avg_latency_ms for r in records]
    mean_lat = _safe_mean(lat)
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    return SequenceMetrics(
        thc=temporal_hoi_consistency(records),
        haa=hazard_anticipation_accuracy(
            records, lead_frames=lead_frames, hazard_threshold=hazard_threshold,
        ),
        rme=risk_weighted_memory_efficiency(records),
        fps=fps,
        latency_ms=mean_lat,
        detection_map=detection_map_stub(records),
    )


def aggregate_sequences(seqs: list[SequenceMetrics]) -> dict[str, float]:
    """Average sequence metrics across multiple sequences into a single summary dict (THC, HAA, RME, FPS, etc.)."""
    if not seqs:
        return {"THC": 0.0, "HAA": 0.0, "RME": 0.0, "FPS": 0.0, "LatencyMS": 0.0, "mAP": 0.0}
    return {
        "THC": _safe_mean([s.thc for s in seqs]),
        "HAA": _safe_mean([s.haa for s in seqs]),
        "RME": _safe_mean([s.rme for s in seqs]),
        "FPS": _safe_mean([s.fps for s in seqs]),
        "LatencyMS": _safe_mean([s.latency_ms for s in seqs]),
        "mAP": _safe_mean([s.detection_map for s in seqs]),
    }


def plot_failure_heatmap(records: list[dict[str, Any]], output_path: str) -> None:
    vals = []
    for rec in records:
        att = rec.get("attention_allocation", {})
        vals.append(_safe_mean([float(v) for v in att.values()]) if att else 0.0)
    arr = np.array(vals, dtype=np.float32).reshape(1, -1) if vals else np.zeros((1, 1), dtype=np.float32)
    plt.figure(figsize=(12, 2))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="AttentionAllocation")
    plt.yticks([])
    plt.xlabel("Frame")
    plt.title("Hazard Attention Heatmap")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_hoi_trajectory(records: list[dict[str, Any]], output_path: str) -> None:
    frames = []
    actual = []
    predicted = []
    for rec in records:
        frames.append(int(rec.get("frame_id", 0)))
        hois = rec.get("hois", [])
        actual.append(sum(1 for h in hois if not h.get("predicted", False)))
        predicted.append(sum(1 for h in hois if h.get("predicted", False)))
    plt.figure(figsize=(10, 4))
    plt.plot(frames, actual, label="actual_hoi")
    plt.plot(frames, predicted, label="predicted_hoi")
    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.title("Predicted vs Actual HOI Trajectory")
    plt.legend()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
