# Reproducibility Report — v1.0.0

**Date**: 2026-03-04
**Branch**: `release/v1.0.0`

## Environment

| Component    | Version                                |
|--------------|----------------------------------------|
| Python       | 3.13.3                                 |
| PyTorch      | 2.10.0                                 |
| TorchVision  | 0.25.0                                 |
| Transformers | 5.2.0                                  |
| ONNX Runtime | 1.24.2                                 |
| NumPy        | 2.4.2                                  |
| Matplotlib   | 3.10.8                                 |
| OS           | macOS 26.3 (arm64, Apple Silicon)      |
| Backend      | MPS (no CUDA)                          |

## Procedure

1. Created a fresh virtual environment (`.venv_repro`) separate from the
   development environment.
2. Installed all dependencies from `requirements_lock.txt` (pinned versions).
3. Installed the project in editable mode: `pip install -e ".[dev,open_vocab]"`.
4. Ran the full test suite.
5. Validated configuration loading.
6. Ran ablations with seed=42 and compared results to `artifacts/v1.0.0/`.

## Test Suite

```
162 passed, 54 warnings in 75.92s
```

All 162 tests pass: unit tests (config, metrics, ablation, memory, HOI,
perception, hazard reasoner, training/export), smoke tests (realtime
pipeline), and stress tests (occlusion, saturation, edge cases).

**Result: PASS**

## Configuration Validation

```
Config validation: PASS
```

`load_config("configs/default.yaml")` loads successfully and passes all
schema checks.

**Result: PASS**

## Ablation Reproduction

Seed-42 ablation results compared against `artifacts/v1.0.0/ablation_results.json`:

| Config         | Metric | Artifact | Repro  | Match |
|----------------|--------|----------|--------|-------|
| baseline       | THC    | 0.1668   | 0.1668 | PASS  |
| baseline       | HAA    | 1.0000   | 1.0000 | PASS  |
| baseline       | RME    | 0.2172   | 0.2172 | PASS  |
| naive_memory   | THC    | 0.1299   | 0.1299 | PASS  |
| naive_memory   | HAA    | 1.0000   | 1.0000 | PASS  |
| naive_memory   | RME    | 0.2172   | 0.2172 | PASS  |

All metrics match within tolerance (< 0.01 absolute).

**Result: PASS**

## Conclusion

The v1.0.0 release is fully reproducible from a clean environment using
`requirements_lock.txt`. Test suite, configuration validation, and ablation
results all reproduce exactly.
