"""Publication-quality plots and tables for ablation study results.

All plots use a consistent style suitable for academic papers: serif fonts,
tight layouts, 300 DPI PNG output, and muted colour palettes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from risksense_vla.eval.ablation import AblationResult

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

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
_METRIC_KEYS = ["THC", "HAA", "RME", "mAP", "FPS"]
_QUALITY_METRICS = ["THC", "HAA", "RME", "mAP"]


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _extract_rows(results: list[AblationResult] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept either AblationResult objects or raw dicts."""
    rows: list[dict[str, Any]] = []
    for r in results:
        if hasattr(r, "as_flat_dict"):
            rows.append(r.as_flat_dict())
        elif isinstance(r, dict):
            rows.append(r)
        else:
            raise TypeError(f"Expected AblationResult or dict, got {type(r)}")
    return rows


# ---------------------------------------------------------------------------
# Grouped bar chart for a single metric
# ---------------------------------------------------------------------------


def plot_metric_comparison(
    results: list[AblationResult] | list[dict[str, Any]],
    metric: str,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Grouped bar chart comparing *metric* across all ablation configs."""
    rows = _extract_rows(results)
    names = [r["ablation"] for r in rows]
    values = [float(r.get(metric, 0.0)) for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    x = np.arange(len(names))
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]
    bars = ax.bar(x, values, color=colours, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.3f}" if metric != "FPS" else f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} across ablation configurations")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# All-metrics grouped bar chart
# ---------------------------------------------------------------------------


def plot_all_metrics(
    results: list[AblationResult] | list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Multi-group bar chart with every metric side by side per config."""
    rows = _extract_rows(results)
    names = [r["ablation"] for r in rows]
    metrics = _QUALITY_METRICS

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.6), 5))
    x = np.arange(len(names))
    width = 0.8 / len(metrics)

    for i, m in enumerate(metrics):
        vals = [float(r.get(m, 0.0)) for r in rows]
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=m, color=_PALETTE[i % len(_PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Quality metrics across ablation configurations")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# FPS vs accuracy scatter
# ---------------------------------------------------------------------------


def plot_fps_vs_accuracy(
    results: list[AblationResult] | list[dict[str, Any]],
    output_path: str | Path,
    *,
    accuracy_metric: str = "THC",
) -> None:
    """Scatter plot with FPS on x-axis, an accuracy metric on y-axis."""
    rows = _extract_rows(results)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, r in enumerate(rows):
        fps = float(r.get("FPS", 0.0))
        acc = float(r.get(accuracy_metric, 0.0))
        colour = _PALETTE[i % len(_PALETTE)]
        ax.scatter(fps, acc, s=80, c=colour, edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(
            r["ablation"], (fps, acc),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color=colour,
        )

    ax.set_xlabel("FPS")
    ax.set_ylabel(accuracy_metric)
    ax.set_title(f"FPS vs {accuracy_metric} trade-off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quantization trade-off
# ---------------------------------------------------------------------------


def plot_quantization_tradeoff(
    results: list[AblationResult] | list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Paired bars showing FPS gain vs quality degradation for quant modes."""
    rows = _extract_rows(results)
    baseline = None
    for r in rows:
        if r["ablation"] == "baseline":
            baseline = r
            break
    if baseline is None and rows:
        baseline = rows[0]

    quant_rows = [r for r in rows if r.get("quant_mode", "fp32") != "fp32"]
    if not quant_rows:
        quant_rows = rows[1:] if len(rows) > 1 else rows

    names = [r["ablation"] for r in quant_rows]
    base_fps = float(baseline["FPS"]) if baseline else 1.0
    base_thc = float(baseline["THC"]) if baseline else 1.0

    fps_deltas = []
    thc_deltas = []
    for r in quant_rows:
        fps_deltas.append(((float(r["FPS"]) - base_fps) / max(abs(base_fps), 1e-6)) * 100.0)
        thc_deltas.append(((float(r["THC"]) - base_thc) / max(abs(base_thc), 1e-6)) * 100.0)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4.5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, fps_deltas, w, label="FPS change (%)", color=_PALETTE[0])
    ax.bar(x + w / 2, thc_deltas, w, label="THC change (%)", color=_PALETTE[1])
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("% change vs baseline")
    ax.set_title("Quantization: FPS gain vs quality impact")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Radar / spider chart
# ---------------------------------------------------------------------------


def plot_radar_chart(
    results: list[AblationResult] | list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Radar chart of all quality metrics normalized to the baseline."""
    rows = _extract_rows(results)
    baseline = None
    for r in rows:
        if r["ablation"] == "baseline":
            baseline = r
            break
    if baseline is None and rows:
        baseline = rows[0]

    metrics = _QUALITY_METRICS + ["FPS"]
    base_vals = [max(float(baseline.get(m, 0.0)), 1e-6) for m in metrics]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for i, r in enumerate(rows):
        vals = [float(r.get(m, 0.0)) / b for m, b in zip(metrics, base_vals)]
        vals.append(vals[0])
        colour = _PALETTE[i % len(_PALETTE)]
        ax.plot(angles, vals, "o-", linewidth=1.5, markersize=4, label=r["ablation"], color=colour)
        ax.fill(angles, vals, alpha=0.08, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Normalized metrics (1.0 = baseline)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def generate_latex_table(
    results: list[AblationResult] | list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """Write a LaTeX-formatted results table suitable for a Q1 paper."""
    rows = _extract_rows(results)
    columns = ["Configuration", "Memory", "HOI", "Attention", "Quant", "THC", "HAA", "RME", "mAP", "FPS"]
    col_fmt = "l" + "c" * (len(columns) - 1)

    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study results across VLA system configurations.}")
    lines.append("\\label{tab:ablation}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\midrule")

    baseline_vals: dict[str, float] = {}
    for r in rows:
        if r["ablation"] == "baseline":
            for m in ("THC", "HAA", "RME", "mAP", "FPS"):
                baseline_vals[m] = float(r.get(m, 0.0))
            break

    for r in rows:
        is_best: dict[str, bool] = {}
        for m in ("THC", "HAA", "RME", "mAP", "FPS"):
            all_vals = [float(row.get(m, 0.0)) for row in rows]
            is_best[m] = float(r.get(m, 0.0)) == max(all_vals) if all_vals else False

        def _fmt(val: float, metric: str) -> str:
            s = f"{val:.3f}" if metric != "FPS" else f"{val:.1f}"
            return f"\\textbf{{{s}}}" if is_best.get(metric) else s

        cells = [
            r["ablation"].replace("_", "\\_"),
            r.get("memory_mode", "").replace("_", "\\_"),
            r.get("hoi_mode", "").replace("_", "\\_"),
            r.get("attention_mode", "").replace("_", "\\_"),
            r.get("quant_mode", "").replace("_", "\\_"),
            _fmt(float(r.get("THC", 0)), "THC"),
            _fmt(float(r.get("HAA", 0)), "HAA"),
            _fmt(float(r.get("RME", 0)), "RME"),
            _fmt(float(r.get("mAP", 0)), "mAP"),
            _fmt(float(r.get("FPS", 0)), "FPS"),
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)
    out = _ensure_dir(output_path)
    out.write_text(table_str, encoding="utf-8")
    return table_str


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_all_plots(
    results: list[AblationResult] | list[dict[str, Any]],
    output_dir: str | Path,
) -> list[str]:
    """Generate all standard ablation plots and the LaTeX table.

    Returns the list of generated file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    for metric in _METRIC_KEYS:
        p = out / f"{metric.lower()}_comparison.png"
        plot_metric_comparison(results, metric, p)
        generated.append(str(p))

    p = out / "all_metrics.png"
    plot_all_metrics(results, p)
    generated.append(str(p))

    p = out / "fps_vs_thc.png"
    plot_fps_vs_accuracy(results, p, accuracy_metric="THC")
    generated.append(str(p))

    p = out / "fps_vs_haa.png"
    plot_fps_vs_accuracy(results, p, accuracy_metric="HAA")
    generated.append(str(p))

    p = out / "quantization_tradeoff.png"
    plot_quantization_tradeoff(results, p)
    generated.append(str(p))

    p = out / "radar_chart.png"
    plot_radar_chart(results, p)
    generated.append(str(p))

    p = out / "summary_table.tex"
    generate_latex_table(results, p)
    generated.append(str(p))

    return generated


# ---------------------------------------------------------------------------
# Hazard timeline visualization
# ---------------------------------------------------------------------------


def plot_hazard_timeline(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Plot hazard scores, attention allocation, and THC drop indicators over time."""
    frames = []
    max_hazards = []
    avg_attentions = []
    thc_actions: list[str | None] = []

    for rec in records:
        frames.append(int(rec.get("frame_id", 0)))
        hazards = rec.get("hazards", [])
        max_hz = max([float(h.get("score", 0.0)) for h in hazards], default=0.0)
        max_hazards.append(max_hz)
        att = rec.get("attention_allocation", {})
        avg_att = float(np.mean(list(att.values()))) if att else 0.0
        avg_attentions.append(avg_att)
        hois = [h for h in rec.get("hois", []) if not h.get("predicted", False)]
        if hois:
            top = max(hois, key=lambda h: float(h.get("confidence", 0.0)))
            thc_actions.append(top.get("action", ""))
        else:
            thc_actions.append(None)

    thc_drops = []
    for i in range(1, len(thc_actions)):
        if thc_actions[i] is not None and thc_actions[i - 1] is not None:
            if thc_actions[i] != thc_actions[i - 1]:
                thc_drops.append(frames[i])

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].fill_between(frames, max_hazards, alpha=0.3, color=_PALETTE[3])
    axes[0].plot(frames, max_hazards, color=_PALETTE[3], linewidth=1)
    axes[0].set_ylabel("Max Hazard")
    axes[0].set_title("Hazard Timeline")

    axes[1].plot(frames, avg_attentions, color=_PALETTE[0], linewidth=1)
    axes[1].set_ylabel("Avg Attention")

    for drop_frame in thc_drops:
        axes[2].axvline(x=drop_frame, color=_PALETTE[1], alpha=0.5, linewidth=0.8)
    axes[2].set_ylabel("THC Drops")
    axes[2].set_xlabel("Frame")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = _ensure_dir(output_path)
    fig.savefig(out)
    plt.close(fig)


def detect_failure_frames(
    records: list[dict[str, Any]],
    hazard_spike_threshold: float = 0.8,
    thc_drop_window: int = 5,
) -> list[dict[str, Any]]:
    """Identify frames exhibiting potential failure indicators.

    Returns a list of dicts with ``frame_id``, ``reason``, and contextual
    data for each detected failure frame.
    """
    failures: list[dict[str, Any]] = []

    prev_action: str | None = None
    consecutive_drops = 0
    for rec in records:
        fid = int(rec.get("frame_id", 0))
        hazards = rec.get("hazards", [])
        max_hz = max([float(h.get("score", 0.0)) for h in hazards], default=0.0)

        if max_hz >= hazard_spike_threshold:
            failures.append({
                "frame_id": fid,
                "reason": "hazard_spike",
                "max_hazard": max_hz,
            })

        hois = [h for h in rec.get("hois", []) if not h.get("predicted", False)]
        if hois:
            top = max(hois, key=lambda h: float(h.get("confidence", 0.0)))
            cur_action = top.get("action", "")
            if prev_action is not None and cur_action != prev_action:
                consecutive_drops += 1
            else:
                consecutive_drops = 0
            prev_action = cur_action
        else:
            consecutive_drops = 0
            prev_action = None

        if consecutive_drops >= thc_drop_window:
            failures.append({
                "frame_id": fid,
                "reason": "sustained_thc_drop",
                "consecutive_action_changes": consecutive_drops,
            })

    return failures
