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

## Proposed Metrics

- THC: Temporal HOI Consistency
- HAA: Hazard Anticipation Accuracy
- RME: Risk-Weighted Memory Efficiency
- Detection mAP
- FPS and latency

## Ablation Template


| Ablation | Hazard Memory | Predictive HOI | Semantic Attention | Quant Bits | THC | HAA | RME | FPS | Notes                 |
| -------- | ------------- | -------------- | ------------------ | ---------- | --- | --- | --- | --- | --------------------- |
| A1       | On            | On             | On                 | 8          |     |     |     |     | Baseline              |
| A1       | Off (naive)   | On             | On                 | 8          |     |     |     |     | Memory effect         |
| A2       | On            | Off            | On                 | 8          |     |     |     |     | Predictive HOI effect |
| A3       | On            | On             | Off (uniform)      | 8          |     |     |     |     | Attention effect      |
| A4       | On            | On             | On                 | 4          |     |     |     |     | INT4 + masking        |


## Runtime Comparison Template


| Device                | Backend       | Quant Bits | Avg FPS | P95 Latency (ms) | THC | HAA | RME |
| --------------------- | ------------- | ---------- | ------- | ---------------- | --- | --- | --- |
| Apple M2              | MPS           | 8          |         |                  |     |     |     |
| RTX 3060              | CUDA/TensorRT | 4          |         |                  |     |     |     |
| Jetson/ARM (optional) | TensorRT      | 4          |         |                  |     |     |     |


## Failure Analysis Visuals

- Hazard attention heatmaps (`outputs/plots/hazard_attention_heatmap.png`)
- Predicted vs actual HOI trajectories (`outputs/plots/hoi_trajectory.png`)
- Temporal consistency trend plots (derived from THC across sequences)

## Optional Extensions

- Synthetic-to-real transfer validation on a small real rare-hazard set.
- Multi-camera integration benchmarks.
- Predictive control hooks for robot/simulator downstream tasks.

