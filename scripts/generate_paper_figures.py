#!/usr/bin/env python3
"""Generate publication-quality figures for the paper.

Reads ablation results (single-seed and multi-seed) and produces PDF
figures in ``paper/figures/`` using the existing plotting infrastructure
plus custom multi-seed bar charts with error bars.

Usage
-----
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --results-dir outputs/ablations --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from risksense_vla.eval.ablation import seed_everything
from risksense_vla.eval.plotting import (
    plot_fps_vs_accuracy,
    plot_quantization_tradeoff,
    plot_radar_chart,
    plot_hazard_timeline,
    detect_failure_frames,
    _PALETTE,
)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOG = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _plot_metric_with_errorbars(
    ms_data: list[dict],
    metric: str,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Bar chart for a single metric with mean +/- std error bars."""
    names = [d["ablation"] for d in ms_data]
    means = [float(d.get(f"{metric}_mean", 0.0)) for d in ms_data]
    stds = [float(d.get(f"{metric}_std", 0.0)) for d in ms_data]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    x = np.arange(len(names))
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]

    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color=colours, edgecolor="white", linewidth=0.5,
        error_kw={"linewidth": 1.2, "capthick": 1.2},
    )

    for bar, val, std in zip(bars, means, stds):
        label = f"{val:.3f}" if metric != "FPS" else f"{val:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(means) * 0.01,
            label, ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} across ablation configurations (mean ± std)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path))
    plt.close(fig)


def _generate_synthetic_timeline(n_frames: int = 150, seed: int = 42) -> list[dict]:
    """Build a synthetic hazard timeline for demonstration plots."""
    seed_everything(seed)
    rng = np.random.RandomState(seed)
    records = []
    actions = ["reach", "grasp", "lift", "place", "idle"]

    for i in range(n_frames):
        t = i / n_frames
        hazard_base = 0.3 + 0.5 * np.sin(2 * np.pi * t) ** 2
        hazard = float(np.clip(hazard_base + rng.normal(0, 0.05), 0, 1))

        n_det = rng.randint(1, 5)
        att = {f"obj_{j}": float(rng.dirichlet(np.ones(n_det))[j]) for j in range(n_det)}
        action = actions[rng.randint(0, len(actions))]
        records.append({
            "frame_id": i,
            "hazards": [{"score": hazard, "label": "knife"}],
            "attention_allocation": att,
            "hois": [{"action": action, "confidence": 0.8 + rng.rand() * 0.2, "predicted": False}],
        })
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper figures.")
    p.add_argument("--results-dir", default="outputs/ablations")
    p.add_argument("--output-dir", default="paper/figures")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ms_path = results_dir / "ablation_results_multiseed.json"
    single_path = results_dir / "ablation_results.json"

    if ms_path.exists():
        ms_data = json.loads(ms_path.read_text())
    else:
        _LOG.error("Multi-seed results not found at %s", ms_path)
        sys.exit(1)

    if single_path.exists():
        single_data = json.loads(single_path.read_text())
    else:
        _LOG.error("Single-seed results not found at %s", single_path)
        sys.exit(1)

    generated = []

    # 1. THC bar chart (mean ± std)
    p1 = out / "thc_comparison.pdf"
    _plot_metric_with_errorbars(ms_data, "THC", p1, title="Temporal HOI Consistency (THC)")
    generated.append(p1)
    _LOG.info("Generated %s", p1)

    # 2. HAA bar chart
    p2 = out / "haa_comparison.pdf"
    _plot_metric_with_errorbars(ms_data, "HAA", p2, title="Hazard Anticipation Accuracy (HAA)")
    generated.append(p2)
    _LOG.info("Generated %s", p2)

    # 3. RME comparison
    p3 = out / "rme_comparison.pdf"
    _plot_metric_with_errorbars(ms_data, "RME", p3, title="Risk-weighted Memory Efficiency (RME)")
    generated.append(p3)
    _LOG.info("Generated %s", p3)

    # 4. FPS vs Accuracy scatter
    p4 = out / "fps_vs_thc.pdf"
    plot_fps_vs_accuracy(single_data, p4, accuracy_metric="THC")
    generated.append(p4)
    _LOG.info("Generated %s", p4)

    # 5. Quantization tradeoff
    p5 = out / "quantization_tradeoff.pdf"
    plot_quantization_tradeoff(single_data, p5)
    generated.append(p5)
    _LOG.info("Generated %s", p5)

    # 6. Radar chart
    p6 = out / "radar_chart.pdf"
    plot_radar_chart(single_data, p6)
    generated.append(p6)
    _LOG.info("Generated %s", p6)

    # 7. Hazard timeline example
    records = _generate_synthetic_timeline(n_frames=150, seed=42)
    p7 = out / "hazard_timeline.pdf"
    plot_hazard_timeline(records, p7)
    generated.append(p7)
    _LOG.info("Generated %s", p7)

    # 8. Failure case example
    failures = detect_failure_frames(records)
    p8 = out / "failure_cases.pdf"
    _plot_failure_cases(records, failures, p8)
    generated.append(p8)
    _LOG.info("Generated %s", p8)

    _LOG.info("All %d figures generated in %s", len(generated), out)


def _plot_failure_cases(
    records: list[dict],
    failures: list[dict],
    output_path: Path,
) -> None:
    """Plot detected failure frames overlaid on the hazard timeline."""
    frames = [r["frame_id"] for r in records]
    hazards = [max((h["score"] for h in r["hazards"]), default=0.0) for r in records]

    spike_frames = [f["frame_id"] for f in failures if f["reason"] == "hazard_spike"]
    drop_frames = [f["frame_id"] for f in failures if f["reason"] == "sustained_thc_drop"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frames, hazards, color=_PALETTE[0], linewidth=1, label="Max hazard")
    ax.fill_between(frames, hazards, alpha=0.15, color=_PALETTE[0])

    if spike_frames:
        spike_y = [hazards[f] for f in spike_frames]
        ax.scatter(spike_frames, spike_y, color=_PALETTE[3], s=40, zorder=5,
                   marker="v", label="Hazard spike")

    for df in drop_frames:
        ax.axvline(x=df, color=_PALETTE[1], alpha=0.4, linewidth=0.8,
                   linestyle="--")
    if drop_frames:
        ax.axvline(x=drop_frames[0], color=_PALETTE[1], alpha=0.4,
                   linewidth=0.8, linestyle="--", label="Sustained THC drop")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Max Hazard Score")
    ax.set_title("Failure Case Detection: Hazard Spikes and THC Drops")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path))
    plt.close(fig)


if __name__ == "__main__":
    main()
