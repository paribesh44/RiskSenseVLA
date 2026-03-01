# HAPVLA

HAPVLA is a modular, edge-deployable Vision-Language-Action system for:

- open-vocabulary perception
- hazard-aware temporal memory
- zero-shot and predictive HOI reasoning
- semantic attention for real-time edge performance

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,open_vocab]"
python scripts/run_realtime.py --config configs/default.yaml
```

If you already have a project venv (for example `.venv311`), activate that instead.

## Perception Backends

The perception tier supports pluggable open-vocabulary detector backends configured in `configs/*.yaml`:

- `grounding_dino` (Transformers-based adapter)
- `yoloe26` (Ultralytics adapter)
- `mock` (fast fallback for local smoke tests)

The embedding backend is also configurable:

- `clip`
- `clip_or_fallback`
- `fallback`

Default behavior is `grounding_dino` + `clip_or_fallback` with graceful CPU fallback if model loading fails.

## Perception-only Validation

Run the perception module independently from memory/HOI/hazard:

```bash
python scripts/run_perception_smoke.py --config configs/default.yaml --max-frames 120
```

Benchmark perception latency/FPS:

```bash
python scripts/eval_perception_fps.py --config configs/default.yaml --mode synthetic --bench-frames 120
```

Outputs:

- smoke log: `outputs/perception_smoke.jsonl`
- FPS report: `outputs/perception_fps.json`

## Novel Contributions

1. Risk-weighted temporal memory that persists hazardous entities longer.
2. Predictive HOI embeddings with 1-3 second anticipation.
3. Semantic attention scheduling to prioritize high-risk regions.
4. Hazard-aware metrics: THC, HAA, and RME.

