# Reproducibility Guide

This document describes how to reproduce the experimental results and ablations for RiskSense-VLA.

## Environment Requirements

- **Python**: 3.11 or newer
- **PyTorch**: 2.1 or newer
- **Other dependencies**: See `pyproject.toml`

For exact reproducibility, capture a full environment snapshot:

```bash
pip freeze > requirements-frozen.txt
```

Include this file when reporting results. Key versions to document:

- Python version
- PyTorch version
- CUDA/cuDNN version (if using GPU)
- Operating system

## Random Seed Control

The project uses `seed_everything()` from `risksense_vla.eval` to set all RNG seeds:

- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `PYTHONHASHSEED` environment variable

For fully deterministic runs, set `PYTHONHASHSEED` before launching:

```bash
export PYTHONHASHSEED=42
python scripts/run_ablations.py --seed 42
```

## Step-by-Step Reproduction

### a) Install

```bash
pip install -e ".[dev,open_vocab]"
```

### b) Generate synthetic data

```bash
python scripts/generate_synthetic_hazards.py --seed 42
```

Output is written to `data/synthetic/` by default.

### c) Export training data

```bash
python scripts/export_synthetic_to_training.py
```

Uses `data/synthetic` as input and writes to `data/training/` by default. Uses `--seed 42` internally for train/val split.

### d) Train all modules

```bash
python scripts/train_all.py --epochs 50
```

Trains perception, HOI, and hazard modules sequentially. Checkpoints are saved to `trained_models/`.

### e) Run ablations

```bash
python scripts/run_ablations.py --seeds 42 123 456 789 1024 --compute-significance
```

Runs all registered ablations across multiple seeds and computes paired t-tests and Cohen's d vs baseline.

### f) Generate plots

Plots are written to `outputs/ablations/plots/` by default. Check that directory after running ablations.

## Hardware Reporting

When reporting results, document:

- **GPU model** (e.g., NVIDIA A100, RTX 4090)
- **Driver version**
- **PyTorch version** (`python -c "import torch; print(torch.__version__)"`)
- **CUDA version** (if applicable)

Example:

```
GPU: NVIDIA A100 40GB
Driver: 535.129.03
PyTorch: 2.1.2+cu121
CUDA: 12.1
```

## Statistical Methodology

- **Multi-seed runs**: Use multiple seeds (e.g., 42, 123, 456, 789, 1024) to estimate variance.
- **95% confidence intervals**: Computed via t-distribution with n-1 degrees of freedom.
- **Paired t-test**: Used to compare ablation variants vs baseline when using the same seeds.
- **Cohen's d**: Effect size metric for practical significance; computed when `--compute-significance` is passed.
