# RiskSense-VLA Experiments

## Reproducibility

All training/eval scripts accept YAML config files:

- `scripts/train_perception.py --config configs/default.yaml --backend-config configs/backend_mps.yaml`
- `scripts/train_hoi.py --config configs/default.yaml --backend-config configs/backend_mps.yaml`
- `scripts/train_hazard_vlm.py --config configs/default.yaml --backend-config configs/backend_mps.yaml`
- `scripts/eval_all.py --log-jsonl outputs/realtime_log.jsonl`

Backend configs expose quantization/pruning/attention knobs:

- `optimization.quant_bits`
- `optimization.pruning_ratio`
- `attention.semantic_attention_threshold`

## Phase 3 Predictive HOI Reproducibility

Phase 3 focuses on predictive HOI only (no visualization/hazard reasoning requirements for baseline runs).

### Dataset Modes

- Raw mode (`--dataset-mode raw`):
  - `--dataset-name hoigen` with `HOIGenRawDataset`
  - `--dataset-name hico` with `HICODetRawDataset`
  - requires `--annotation-json`
- Preprocessed mode (`--dataset-mode preprocessed`):
  - `TemporalHOIPreprocessedDataset`
  - requires `--preprocessed-jsonl`
  - optional `--val-preprocessed-jsonl`

### Preprocessing

Convert HOIGen/HICO-like annotations to temporal JSONL:

```bash
python scripts/preprocess_hoi.py \
  --input-json data/hoi/raw_annotations.json \
  --output-jsonl data/hoi/temporal_train.jsonl \
  --window 4
```

### Train / Fine-tune `PredictiveHOINet`

Raw HOIGen example:

```bash
python scripts/train_hoi.py \
  --config configs/default.yaml \
  --backend-config configs/backend_mps.yaml \
  --dataset-mode raw \
  --dataset-name hoigen \
  --annotation-json data/hoigen/train.json \
  --epochs 5 \
  --batch-size 32 \
  --output artifacts/hoi_hoigen.pt
```

Preprocessed example:

```bash
python scripts/train_hoi.py \
  --config configs/default.yaml \
  --backend-config configs/backend_mps.yaml \
  --dataset-mode preprocessed \
  --preprocessed-jsonl data/hoi/temporal_train.jsonl \
  --val-preprocessed-jsonl data/hoi/temporal_val.jsonl \
  --epochs 5 \
  --batch-size 32 \
  --output artifacts/hoi_preprocessed.pt
```

### Run Predictive HOI Inference Logging

```bash
python scripts/run_hoi_inference.py \
  --config configs/default.yaml \
  --max-frames 200 \
  --checkpoint artifacts/hoi_preprocessed.pt \
  --log-jsonl outputs/hoi_inference.jsonl
```

Logged fields include:

- `hoi_current`
- `hoi_future_embeddings`
- `hoi_future_action_labels`
- `latency_ms` (perception/memory/hoi/total)

### Evaluate Predictive Accuracy

With optional ground-truth JSONL:

```bash
python scripts/eval_hoi.py \
  --pred-log-jsonl outputs/hoi_inference.jsonl \
  --gt-jsonl data/hoi/temporal_val.jsonl \
  --report-json outputs/hoi_eval.json
```

## Phase 3 Metrics

Tracked for predictive HOI:

- `current_top1` / `current_action_top1` (current action correctness)
- `future_top1` / `future_action_top1_by_horizon` (1s/2s/3s)
- `future_embedding_cosine` / `future_embedding_cosine_by_horizon`
- `fps`
- `latency_ms`

These metrics complement global VLA metrics and are used as Phase 3 acceptance checks before deeper hazard integration.

## Phase 3 Testing and QA

Unit tests:

- `tests/unit/test_hoi_module.py`
- `tests/unit/test_hoi_predictor.py`

Smoke integration:

- `tests/smoke/test_realtime_pipeline.py` (`test_smoke_predictive_hoi_infer`)

Recommended run:

```bash
python3 -m pytest tests/unit/test_hoi_module.py tests/unit/test_hoi_predictor.py tests/smoke/test_realtime_pipeline.py
```

Temporal coherence is validated by sequential-frame assertions in `test_hoi_module.py`.

## Phase 3 -> Phase 4 Integration Point

`HOIInferenceOutput.as_triplets()` provides a direct bridge from predictive HOI output to hazard reasoning inputs without changing Phase 3 training/inference codepaths.

## Metrics Definitions

### THC: Temporal HOI Consistency

Proportion of consecutive frames where the top observed HOI action remains unchanged.

```
THC = (1 / (T-1)) * sum_{t=2}^{T} 1[a_t* = a_{t-1}*]
```

where `a_t*` is the highest-confidence observed action at frame `t`. Frames without observed HOIs are skipped.

### HAA: Hazard Anticipation Accuracy

Fraction of hazardous events that were preceded by a predicted HOI within a lead window.

```
HAA = (1 / |H|) * sum_{h in H} 1[exists p in P : h - L <= p <= h]
```

where `H` = set of frames with hazard score >= threshold (default 0.7), `P` = frames with predicted HOIs, `L` = lead window (default 25 frames).

### RME: Risk-weighted Memory Efficiency

Alignment between compute allocation and hazard intensity.

```
RME = clip(mean(s_t * c_t), 0, 1)
```

where `s_t` is mean hazard score and `c_t` is mean attention allocation at frame `t`.

### FPS and Latency

FPS is derived from per-frame latency: `FPS = 1000 / mean_latency_ms`. Latency is broken down into perception, memory, HOI, and hazard reasoning components.

### detection mAP (stub)

Currently a placeholder using mean detection confidence. Will be replaced with proper mAP once a ground-truth benchmark adapter is connected.

## Ablation Configurations

Seven registered ablation configs are available:

| Name | Memory | HOI | Attention | Quantization | Description |
|------|--------|-----|-----------|--------------|-------------|
| `baseline` | hazard_aware | predictive | semantic | fp32 | Full system |
| `naive_memory` | naive | predictive | semantic | fp32 | Uniform-decay memory |
| `frame_only_hoi` | hazard_aware | frame_only | semantic | fp32 | No future prediction |
| `uniform_attention` | hazard_aware | predictive | uniform | fp32 | Equal compute allocation |
| `int8_qat` | hazard_aware | predictive | semantic | int8 | INT8 QAT |
| `int4_ptq` | hazard_aware | predictive | semantic | int4_ptq | INT4-style PTQ |
| `int8_masked` | hazard_aware | predictive | semantic | int8_masked | INT8 + pruning |

## Running Ablations

```bash
# All ablations, single seed
python scripts/run_ablations.py --config configs/default.yaml

# Multi-seed with significance testing
python scripts/run_ablations.py --seeds 42 123 456 789 1024 --compute-significance

# Specific ablations only
python scripts/run_ablations.py --ablations baseline naive_memory int8_qat
```

## Statistical Methodology

### Multi-seed Evaluation

Each ablation is run across N seeds (recommended N >= 5). For each metric:
- **Mean and standard deviation** are computed across seeds
- **95% confidence intervals** use the t-distribution: `CI = mean +/- t_{0.975, N-1} * std / sqrt(N)`
- **Paired t-test** compares each variant against baseline (same seeds)
- **Cohen's d effect size** quantifies practical significance: `d = mean(diff) / std(diff)`

### Warmup Exclusion

Module benchmarks exclude the first `--warmup` iterations (default 50) to avoid cold-start effects. GPU synchronization (`torch.cuda.synchronize()`) is called before timing on CUDA devices.

### Reproducibility

- `seed_everything()` sets: `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `PYTHONHASHSEED`, `torch.use_deterministic_algorithms(True, warn_only=True)`
- Synthetic sequences use `np.random.RandomState(seed)` for per-sequence determinism

## Hardware Reporting

When reporting results, document:
- CPU model and core count
- GPU model, VRAM, driver version (if applicable)
- PyTorch version and CUDA/MPS backend version
- OS and Python version
- Peak memory usage (logged per training epoch)

## Runtime Comparison Template

| Device | Backend | Quant | Avg FPS | P95 Latency (ms) | THC | HAA | RME |
|--------|---------|-------|---------|-------------------|-----|-----|-----|
| Apple M2 | MPS | INT8 | | | | | |
| RTX 3060 | CUDA/TensorRT | INT4 | | | | | |
| Jetson/ARM | TensorRT | INT4 | | | | | |

## Failure Analysis

### Known Failure Modes

1. **Fast occlusion**: Objects disappearing/reappearing every 2-3 frames can cause THC drops as action labels flicker.
2. **Multi-hazard saturation**: With 10+ simultaneous high-risk objects, attention allocation becomes nearly uniform, reducing RME benefit.
3. **Object disappearance**: When all objects vanish simultaneously, memory state decays to zero; re-detection requires fresh persistence buildup.
4. **Motion blur**: Severe motion blur degrades detection confidence, causing embedding instability.

### Failure Detection

Use `detect_failure_frames()` from `risksense_vla.eval.plotting` to automatically flag:
- Hazard spikes (score >= 0.8)
- Sustained THC drops (5+ consecutive action changes)

### Visualization

- Hazard timeline: `plot_hazard_timeline(records, "outputs/hazard_timeline.png")`
- Failure heatmap: `plot_failure_heatmap(records, "outputs/failure_heatmap.png")`
- HOI trajectory: `plot_hoi_trajectory(records, "outputs/hoi_trajectory.png")`

## Optional Extensions

- Synthetic-to-real transfer validation on a small real rare-hazard set.
- Multi-camera integration benchmarks.
- Predictive control hooks for robot/simulator downstream tasks.

