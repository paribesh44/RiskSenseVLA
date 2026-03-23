# Experiment Artifacts

This directory contains versioned snapshots of experiment results, configs,
and hardware metadata for each release.

## v1.0.0 (2026-03-04)

Directory: `v1.0.0/`

| File                                | Description                                    |
|-------------------------------------|------------------------------------------------|
| `ablation_results.csv`              | Per-seed ablation results (first seed only)    |
| `ablation_results.json`             | Same data in JSON format                       |
| `ablation_results_multiseed.csv`    | Aggregated multi-seed results (mean +/- std)   |
| `ablation_results_multiseed.json`   | Same data in JSON format                       |
| `significance_tests.json`           | Paired t-test and Cohen's d vs baseline        |
| `hardware_summary.json`             | Hardware/OS/library versions for the run       |
| `configs_snapshot/`                 | Copy of all YAML configs used                  |

### Seeds Used

42, 123, 456, 789, 1024

### Ablation Configurations

baseline, naive_memory, frame_only_hoi, uniform_attention, int8_qat,
int4_ptq, int8_masked

### Metrics Collected

THC, HAA, RME, mAP, FPS, latency_ms, peak_memory_mb
