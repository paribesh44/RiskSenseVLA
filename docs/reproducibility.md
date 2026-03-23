# Environment & Reproducibility

This document provides the exact environment specification for reproducing
RiskSense-VLA v1.0.0 results.

## Verified Environment

| Component      | Version                                |
|----------------|----------------------------------------|
| Python         | 3.13.3                                 |
| PyTorch        | 2.10.0                                 |
| TorchVision    | 0.25.0                                 |
| Transformers   | 5.2.0                                  |
| ONNX Runtime   | 1.24.2                                 |
| NumPy          | 2.4.2                                  |
| Matplotlib     | 3.10.8                                 |
| OS (dev)       | macOS 26.3 (arm64, Apple Silicon / MPS)|
| CUDA           | N/A on dev machine; see GPU notes below|

## Recreating the Environment

### Option A — pip (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_lock.txt
pip install -e ".[dev,open_vocab]"
```

`requirements_lock.txt` pins every transitive dependency to the exact
versions used during the v1.0.0 validation run.

### Option B — conda

```bash
conda env create -f environment.yml
conda activate risksense-vla
pip install -e ".[dev,open_vocab]"
```

## GPU / CUDA Notes

The primary development and validation was performed on macOS with Apple
Silicon (MPS backend). For NVIDIA GPU environments:

- Install PyTorch with the appropriate CUDA index:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- Ensure the NVIDIA driver supports CUDA >= 12.1.
- Install the optional `cuda` extras: `pip install -e ".[cuda]"` for TensorRT
  support.

Document your exact setup when reporting results:

```
GPU: <model>
Driver: <version>
CUDA: <version>
cuDNN: <version>
```

## Deterministic Seeding

All experiments use `seed_everything()` from `risksense_vla.eval.ablation`:

```python
from risksense_vla.eval.ablation import seed_everything
seed_everything(42)
```

This sets `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`,
`torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`, and `PYTHONHASHSEED`.

For fully deterministic runs:

```bash
export PYTHONHASHSEED=42
python scripts/run_ablations.py --seeds 42 123 456 789 1024 --compute-significance
```

## Reproduction Steps

1. **Install** — see Option A above.
2. **Generate synthetic data** — `python scripts/generate_synthetic_hazards.py --seed 42`
3. **Export training data** — `python scripts/export_synthetic_to_training.py`
4. **Train all modules** — `python scripts/train_all.py --epochs 50`
5. **Run ablations** — `python scripts/run_ablations.py --seeds 42 123 456 789 1024 --compute-significance`
6. **Run tests** — `pytest tests/ -v`
7. **Generate figures** — `python scripts/generate_paper_figures.py`

## Result Tolerance

Floating-point results may vary across hardware (CPU vs GPU, MPS vs CUDA) and
library versions. Results are considered reproduced if:

- THC, HAA, RME differ by less than 0.01 (absolute).
- FPS varies with hardware; relative ranking between configs should be
  preserved.
