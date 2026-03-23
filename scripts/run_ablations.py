#!/usr/bin/env python3
"""Run ablation studies for the RiskSense-VLA system.

Executes evaluation pipelines across multiple configuration variants
(memory architecture, HOI mode, attention scheduling, quantization),
collects metrics (THC, HAA, RME, mAP, FPS), writes CSV results,
generates publication-quality plots, and prints a summary table.

Usage
-----
    # Run all registered ablations with default settings
    python scripts/run_ablations.py

    # Run specific ablations
    python scripts/run_ablations.py --ablations baseline naive_memory int8_qat

    # Custom output directory and seed
    python scripts/run_ablations.py --output-dir results/exp01 --seed 123

    # Use a YAML ablation config for custom variants
    python scripts/run_ablations.py --ablation-config configs/ablations.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from risksense_vla.config import load_config
from risksense_vla.eval.ablation import (
    ABLATION_REGISTRY,
    AblationConfig,
    AblationRunner,
    compute_significance,
    load_ablation_configs_from_yaml,
    multi_seed_results_to_csv,
    print_summary_table,
    results_to_csv,
)
from risksense_vla.eval.plotting import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ablation studies for the RiskSense-VLA system.",
    )
    p.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to the base YAML config (default: configs/default.yaml)",
    )
    p.add_argument(
        "--backend-config", default=None,
        help="Optional backend override YAML (e.g. configs/backend_cuda.yaml)",
    )
    p.add_argument(
        "--ablation-config", default=None,
        help="Optional YAML file defining ablation variants (overrides the built-in registry)",
    )
    p.add_argument(
        "--dataset-dir", default="outputs",
        help="Directory with benchmark JSONL sequences (falls back to synthetic data)",
    )
    p.add_argument(
        "--trained-dir", default="trained_models",
        help="Directory containing trained model checkpoints",
    )
    p.add_argument(
        "--output-dir", default="outputs/ablations",
        help="Directory for CSV, plots, and summary outputs",
    )
    p.add_argument(
        "--ablations", nargs="+", default=None,
        help="Specific ablation names to run (default: all registered)",
    )
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Multiple seeds for multi-seed runs (e.g. --seeds 42 123 456 789 1024)",
    )
    p.add_argument(
        "--compute-significance", action="store_true",
        help="Compute paired t-test and Cohen's d vs baseline (requires --seeds)",
    )
    p.add_argument("--warmup", type=int, default=50, help="Benchmark warmup iterations")
    p.add_argument("--iterations", type=int, default=200, help="Benchmark measurement iterations")
    p.add_argument(
        "--device", default="cpu",
        help="Device for model benchmarks (cpu, cuda, mps)",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation",
    )
    return p.parse_args()


def _resolve_configs(args: argparse.Namespace) -> list[AblationConfig]:
    """Build the list of ablation configs to run."""
    if args.ablation_config:
        registry = load_ablation_configs_from_yaml(args.ablation_config)
    else:
        registry = dict(ABLATION_REGISTRY)

    if args.ablations:
        selected: list[AblationConfig] = []
        for name in args.ablations:
            if name not in registry:
                _LOG.warning("Unknown ablation '%s'; skipping. Available: %s", name, list(registry.keys()))
                continue
            cfg = registry[name]
            cfg.seed = args.seed
            selected.append(cfg)
        return selected

    configs = list(registry.values())
    for c in configs:
        c.seed = args.seed
    return configs


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config, args.backend_config) if args.backend_config else load_config(args.config)

    configs = _resolve_configs(args)
    if not configs:
        _LOG.error("No ablation configurations to run.")
        sys.exit(1)

    _LOG.info("Running %d ablation(s): %s", len(configs), [c.name for c in configs])

    import torch
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    runner = AblationRunner(
        cfg=cfg,
        dataset_dir=args.dataset_dir,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds and len(args.seeds) > 1:
        ms_results = runner.run_all_multi_seed(configs, args.seeds)

        csv_path = out_dir / "ablation_results_multiseed.csv"
        multi_seed_results_to_csv(ms_results, csv_path)
        _LOG.info("Multi-seed CSV written to %s", csv_path)

        results_json = out_dir / "ablation_results_multiseed.json"
        results_json.write_text(
            json.dumps([r.as_flat_dict() for r in ms_results], indent=2),
            encoding="utf-8",
        )
        _LOG.info("Multi-seed JSON written to %s", results_json)

        if args.compute_significance and len(ms_results) > 1:
            baseline_ms = ms_results[0]
            sig_results = []
            for ms in ms_results[1:]:
                for metric in ("thc", "haa", "rme", "fps"):
                    sig = compute_significance(baseline_ms, ms, metric=metric)
                    sig["ablation"] = ms.config_name
                    sig["metric"] = metric
                    sig_results.append(sig)
            sig_path = out_dir / "significance_tests.json"
            sig_path.write_text(json.dumps(sig_results, indent=2), encoding="utf-8")
            _LOG.info("Significance tests written to %s", sig_path)

        results = [r.per_seed_results[0] for r in ms_results]
    else:
        results = runner.run_all(configs)

    csv_path = out_dir / "ablation_results.csv"
    results_to_csv(results, csv_path)
    _LOG.info("CSV written to %s", csv_path)

    results_json = out_dir / "ablation_results.json"
    results_json.write_text(
        json.dumps([r.as_flat_dict() for r in results], indent=2),
        encoding="utf-8",
    )
    _LOG.info("JSON written to %s", results_json)

    if not args.no_plots:
        plots_dir = out_dir / "plots"
        generated = generate_all_plots(results, plots_dir)
        _LOG.info("Generated %d plots in %s", len(generated), plots_dir)
    else:
        _LOG.info("Plot generation skipped (--no-plots)")

    _LOG.info("\n%s", "=" * 70)
    _LOG.info("ABLATION STUDY RESULTS")
    _LOG.info("%s", "=" * 70)
    print_summary_table(results)
    _LOG.info("%s\n", "=" * 70)


if __name__ == "__main__":
    main()
