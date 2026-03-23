# RiskSense-VLA — Project overview (This Research project is in progress.)

## 1. Introduction

This document is a **standalone, high-level overview** for readers who want, in one pass:

- **Motivation** — why the system exists and which questions it is meant to answer  
- **Contributions** — what the project adds compared with a single end-to-end model or a generic VLA stack  
- **Architecture** — how perception, memory, HOI, hazard reasoning, and attention connect  
- **Outcomes** — what “success” means in engineering and evaluation terms, and **where to find or produce** quantitative results  

For module-specific APIs, diagrams, and command-line recipes, see `[architecture.md](architecture.md)`, `[modules.md](modules.md)`, and `[experiments.md](experiments.md)`. The `[README.md](../README.md)` remains the entry point for install and quickstart.

**Repository:** [github.com/paribesh44/RiskSenseVLA](https://github.com/paribesh44/RiskSenseVLA)

---

## 2. What RiskSense-VLA is

RiskSense-VLA is a **modular Vision–Language–Action (VLA)–style system** for **hazard-aware, real-time scene understanding**. It is built for **laptop-class inference** (Apple Silicon / MPS by default; CUDA and stronger VLMs optional), not only for datacenter-scale batch jobs.

The stack combines:

- **Open-vocabulary perception** (detection, optional masks, fixed-dimensional embeddings)
- **Hazard-weighted temporal memory** (lightweight recurrence over objects and interactions)
- **Predictive human–object interaction (HOI) reasoning** with a **1–3 second** anticipation horizon
- **Prompt-driven hazard reasoning** via **pluggable vision–language backends**
- **Semantic attention scheduling** to bias compute toward high-risk regions

Design goals include **clear module boundaries**, **YAML-driven configuration**, **scripted training and evaluation**, **ablation studies**, and **export** to TorchScript and ONNX for deployment-oriented workflows.

---

## 3. Motivation and problem framing

**Motivation.** Assistive robotics, monitoring, and embodied agents need **stable, interpretable** answers over time—not only a single-frame label. Monolithic VLMs are hard to ablate, expensive at the edge, and weakly structured for safety logging or control. RiskSense-VLA is motivated by the need for a **composable pipeline**: swap detectors or VLMs, tune memory and attention, and measure hazard-relevant behavior with explicit metrics, without rewriting the whole stack.

**Problem framing.** Many applications must answer, over time:

1. **What** objects and regions matter in the scene (possibly described in open vocabulary)?
2. **How** do humans and objects interact now, and **how might** those interactions evolve in the next seconds?
3. **Which** situations warrant alerts or caution, with enough structure for logging, control, or human review?

RiskSense-VLA treats these as a **pipeline** with explicit data contracts between stages, so components can be improved, replaced, or ablated without rewriting the entire system.

---

## 4. End-to-end pipeline

At runtime, a typical flow is:

1. **Input** — Video frames (or synthetic frames) with timestamps.
2. **Perception** — Open-vocabulary detector + embedder produce `list[PerceptionDetection]` (boxes, labels, confidences, optional masks, `clip_embedding`).
3. **Memory** — `HazardAwareMemory` updates a `MemoryState` using a **linear SSM–style** recurrence; hazard signals can **slow decay** for risky tracks.
4. **HOI** — `PredictiveHOIModule` consumes memory + detections and emits current HOIs, future embeddings, and future action hypotheses over the configured horizon.
5. **Hazard** — `HazardReasoner` / `DistilledHazardReasoner` calls a VLM backend to produce per-track hazard scores, alerts, optional explanations, and feedback paths into memory.
6. **Attention** — `SemanticAttentionScheduler` uses hazard maps to schedule emphasis for the next processing cycle.

Module graphs, sequence diagrams, and Phase 3/4 detail are in `[architecture.md](architecture.md)`.

---

## 5. Major components

### 5.1 Perception (`OpenVocabPerception`)

Perception is **router-based**: detector and embedder implementations are chosen from configuration.

- **Detectors** (examples): Grounding DINO, YOLOE26, mock (testing; gated by config).
- **Embedders**: CLIP, CLIP with automatic fallback, or histogram-style fallback embeddings.

Outputs conform to a **canonical contract**: each detection includes `track_id`, `label`, `confidence`, `bbox_xyxy`, optional `mask`, and `clip_embedding`. Downstream code depends on this contract, not on a specific detector brand.

Platform notes (e.g., Grounding DINO on CPU when the runtime device is MPS) are documented in the README.

### 5.2 Temporal memory (`HazardAwareMemory`)

Memory is implemented as a **lightweight linear recurrent update** (fixed projections and gated/hazard-aware retention). It maintains:

- Per-object state suitable for tracking over missing frames
- A compact **HOI-related embedding** and **canonical state vector** for the HOI and hazard modules

APIs support both **stateful** (`HazardAwareMemory.update`) and **functional** (`update_hazard_memory`) use. Optional per-detection hazard lists or `HazardScore` events tune retention.

### 5.3 Predictive HOI (`PredictiveHOIModule`)

The HOI module performs **zero-shot / prototype-style** action association and **short-horizon prediction**:

- **Inputs:** `MemoryState`, `list[PerceptionDetection]`, timestamp, optional `horizon_seconds` (clamped to 1–3).
- **Outputs:** `HOIInferenceOutput` — current HOIs, `hoi_future_embeddings`, future action labels and confidences, plus `as_triplets()` for hazard integration.

Training and dataset modes (raw HOIGen/HICO-style, preprocessed temporal JSONL) are described in `[modules.md](modules.md)` and `[experiments.md](experiments.md)`.

### 5.4 Hazard reasoning (`HazardReasoner`)

Hazard reasoning is **backend-pluggable**:

- Compact VLMs (e.g., SmolVLM-class) for portable deployment
- Larger multimodal models (e.g., Phi-4 multimodal) when CUDA-class memory is available
- Tiny/stub backends for CI and fast wiring checks when `lightweight_mode` is enabled

The reasoner builds **structured prompts** from memory summaries, current HOIs, optional geometry/proximity cues, and optional frames; it parses model outputs into **risk scores**, **maps**, **alerts**, and optional **natural-language explanations**. Runtime JSONL logs can include hazard timing, backend metadata, and optional prompt debug fields.

### 5.5 Semantic attention (`SemanticAttentionScheduler`)

Given hazard maps and detections, the scheduler **prioritizes regions or entities** that are semantically high-risk, improving **effective throughput** on resource-constrained devices without abandoning open-vocabulary perception.

---

## 6. Configuration

- **Primary config:** `configs/default.yaml` — perception, memory, HOI, hazard, attention, runtime.
- **Backend overlays:** e.g. `configs/backend_mps.yaml`, `configs/backend_cuda.yaml` — device, quantization, pruning, attention thresholds.
- **Ablations:** `configs/ablations.yaml` and `scripts/run_ablations.py`.

This keeps **experiments declarative**: the same code paths run with different YAML overlays.

---

## 7. Training, evaluation, and benchmarks

Documented entrypoints include:

- Perception, HOI, and hazard VLM training scripts (see `[experiments.md](experiments.md)`)
- **E2E verification:** `scripts/run_e2e_verify.py` (with `--fast` for lightweight checks)
- **Realtime demo / logging:** `scripts/run_realtime.py`
- **Phase-4 benchmark gate:** `scripts/benchmark_phase4.py` (FPS and requirements such as GPU can be enforced)
- **Ablations:** multi-seed runs, optional significance testing (e.g., paired tests, effect sizes), CSV/JSON outputs and plots under `outputs/ablations/`

Aggregating sequence metrics uses helpers under `risksense_vla.eval` as described in the experiments doc.

---

## 8. Metrics oriented toward hazards

The project emphasizes metrics that align with **temporal consistency**, **anticipation**, and **memory under risk**, for example:

- **THC** — temporal HOI consistency  
- **HAA** — hazard anticipation accuracy  
- **RME** — risk-weighted memory efficiency  

These appear alongside conventional detection-oriented figures where applicable (e.g., mAP) in ablation reporting. Formal definitions, Phase 3 HOI metrics (e.g., future action accuracy by horizon, embedding cosine, FPS), and reporting templates are in `[experiments.md](experiments.md)`.

---

## 9. Expected outcomes and empirical results

**What we aim for (engineering).** A reproducible loop that runs on **resource-constrained** targets: open-vocabulary perception, temporal memory with hazard-aware retention, short-horizon HOI prediction, VLM-backed hazard signals, and attention that shifts compute toward risky regions—**without** requiring a single frozen monolith.

**What we aim for (evaluation).** Measurable behavior along the axes above: predictive HOI quality and latency (Phase 3), end-to-end checks and Phase 4 gates (throughput, optional hardware requirements), and hazard-oriented sequence metrics (THC, HAA, RME) plus standard complements (e.g., mAP where wired). See `[experiments.md](experiments.md)` for definitions and acceptance-style notes.

**Where results live.**

- **Your runs:** JSONL from `scripts/run_realtime.py`, `scripts/run_hoi_inference.py`, and related tools; ablation tables under `outputs/ablations/`.
- **Versioned snapshots:** `[artifacts/README.md](../artifacts/README.md)` describes bundled CSV/JSON (e.g., multi-seed ablations, significance tests, config snapshots) for tagged releases.

**How to report outcomes.** Always tie numbers to **config paths, seeds, hardware, and the exact script invocation**. This overview does **not** publish fixed benchmark scores; use your logs, `artifacts/` snapshots, and any paper or report you maintain for tables and comparisons. If work is **in progress**, treat the bullets above as **targets** and document partial results with the same provenance.

---

## 10. Testing, containers, and export

- **Tests:** `pytest tests/` (including stress tests under `tests/stress/`).
- **Docker:** CPU headless image and Compose setup for reproducible runs without a full local ML stack; see README.
- **Export:** `scripts/export_models.py` — TorchScript and ONNX for selected modules, with optional QAT-related steps.

---

## 11. Code layout (orientation)


| Area                  | Typical location                |
| --------------------- | ------------------------------- |
| Perception            | `src/risksense_vla/perception/` |
| Memory                | `src/risksense_vla/memory/`     |
| HOI                   | `src/risksense_vla/hoi/`        |
| Hazard / VLM backends | `src/risksense_vla/hazard/`     |
| Attention             | `src/risksense_vla/attention/`  |
| Runtime wiring        | `src/risksense_vla/runtime/`    |
| Config loading        | `src/risksense_vla/config/`     |
| Scripts               | `scripts/`                      |
| Tests                 | `tests/`                        |


---

## 12. Contributions (what this project adds)

Relative to a **single large VLM** or a generic VLA without explicit safety structure, the main contributions are:

1. **Risk-weighted temporal memory** — retains representations of hazardous entities longer via gated, hazard-aware decay (see §5.2).
2. **Predictive HOI module** — short-horizon (1–3 s) action hypotheses and embeddings for anticipation-style reasoning (see §5.3).
3. **Semantic attention scheduling** — biases processing toward semantically high-risk regions under a fixed compute budget (see §5.5).
4. **Hazard-aware evaluation metrics** — THC, HAA, and RME to score temporal consistency, anticipation, and risk-weighted memory use (see §8 and `[experiments.md](experiments.md)`).
5. **Open, reproducible software** — YAML-driven configs, pluggable perception and VLM backends, training/eval/ablation scripts, export paths, and containerized CPU workflows (see §6–§7, §10).

Together these are **architectural and objective-level** contributions: composable safety-aware perception and reasoning rather than one end-to-end black box.

---

## 13. Further reading


| Document                             | Contents                                                             |
| ------------------------------------ | -------------------------------------------------------------------- |
| `[README.md](../README.md)`          | Install, quickstart, Docker, perception/memory/hazard usage snippets |
| `[architecture.md](architecture.md)` | Module graph, Phase 3/4 flows, config surfaces, logging              |
| `[modules.md](modules.md)`           | HOI API, datasets, training/inference scripts                        |
| `[experiments.md](experiments.md)`   | Reproducible train/eval commands, metrics definitions, optimization knobs |
| `[artifacts/README.md](../artifacts/README.md)` | Versioned result snapshots and metric columns for releases     |


---

*This overview summarizes design intent and contribution claims. Any quantitative comparison should cite a concrete run, config snapshot, or published artifact—not this page alone.*
