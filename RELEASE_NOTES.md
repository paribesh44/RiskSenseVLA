# Release Notes — v1.0.0

**Release date**: 2026-03-04

## Summary

RiskSense-VLA v1.0.0 is the first stable release of the hazard-aware
vision-language-action system. It delivers a complete, reproducible research
pipeline covering perception, hazard-aware temporal memory, predictive
human-object interaction, semantic attention scheduling, and edge deployment
via quantization.

## Key Contributions

- **Hazard-aware temporal memory** — a linear time-varying SSM where decay,
  input gating, and observation boosts are conditioned on hazard signals,
  providing risk-proportional entity persistence.
- **Predictive HOI embeddings** — projects current interaction states 1-3
  seconds into the future, enabling anticipatory hazard detection.
- **Semantic attention scheduling** — allocates compute proportionally to
  hazard risk for edge-deployable efficiency.
- **Three novel metrics** — Temporal HOI Consistency (THC), Hazard
  Anticipation Accuracy (HAA), and Risk-weighted Memory Efficiency (RME).

## Metrics Validated

| Metric | Description                                   |
|--------|-----------------------------------------------|
| THC    | Temporal HOI Consistency across frames         |
| HAA    | Hazard Anticipation Accuracy (lead-frame window) |
| RME    | Risk-weighted Memory Efficiency                |
| mAP    | Detection mean average precision (stub)        |
| FPS    | Inference throughput on target hardware         |

## Statistical Rigor

- Multi-seed evaluation (seeds 42, 123, 456, 789, 1024) for variance
  estimation.
- 95 % confidence intervals via t-distribution.
- Paired t-tests and Cohen's d for ablation significance vs baseline.
- Deterministic seeding (`seed_everything`) for full reproducibility.

## Ablation Coverage

Seven registered ablation configurations isolating individual architectural
axes:

| Config             | Axis changed       |
|--------------------|---------------------|
| baseline           | (full system)       |
| naive_memory       | memory_mode=naive   |
| frame_only_hoi     | hoi_mode=frame_only |
| uniform_attention  | attention_mode=uniform |
| int8_qat           | quant_mode=int8     |
| int4_ptq           | quant_mode=int4_ptq |
| int8_masked        | quant_mode=int8_masked |

## Stress Testing Coverage

- Rapid occlusion recovery
- Multi-hazard saturation (10+ simultaneous high-risk objects)
- Complete object disappearance
- High detection counts (32 objects)
- Extended zero-detection sequences
- Metric edge cases (empty HOIs, single-frame evaluation)

## Deployment Validation

- INT8 QAT and INT4 PTQ quantization verified.
- TorchScript and ONNX export roundtrip validated.
- Real-time pipeline smoke test (perception -> memory -> HOI -> hazard ->
  attention).
- Hardware benchmark script (`scripts/benchmark_phase4.py`).

## Known Limitations

1. **Synthetic-only evaluation** — all experiments use procedurally generated
   data; real-world validation is future work.
2. **Single-camera assumption** — multi-camera fusion is not evaluated.
3. **Fixed SSM parameters** — hand-tuned, not learned end-to-end.
4. **Placeholder mAP** — detection mAP is a mean-confidence stub; a proper
   ground-truth adapter is needed.
5. **VLM memory footprint** — the Phi-4 backend requires ~10 GB GPU memory;
   the TinyHazardNet fallback trades accuracy for deployability.
6. **No closed-loop control** — the system produces hazard signals but does
   not integrate with a robot controller.
7. **Limited real-world stress testing** — adversarial occlusion and sensor
   failures are not covered.
