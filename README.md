# RiskSense-VLA

RiskSense-VLA is a modular, edge-deployable Vision-Language-Action system for:

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

## Docker (CPU, headless)

If local Python dependencies are unstable, run the project in Docker:

```bash
docker build -t risksense-vla:cpu .
docker run --rm -it \
  -v "$PWD/outputs:/app/outputs" \
  risksense-vla:cpu
```

The default container command runs:

```bash
python scripts/run_realtime.py --config configs/popup_demo.yaml --max-frames 30 --no-display
```

To run a different command:

```bash
docker run --rm -it \
  -v "$PWD/outputs:/app/outputs" \
  risksense-vla:cpu \
  python scripts/run_e2e_verify.py --config configs/default.yaml --fast
```

## Docker Compose

You can also run with Compose:

```bash
docker compose up --build
```

This uses `docker-compose.yml` and runs the same popup demo command in headless mode,
while mounting:

- `./outputs` -> `/app/outputs`
- `~/.cache/huggingface` -> `/root/.cache/huggingface`

Run a one-off command with Compose:

```bash
docker compose run --rm risksense-vla python scripts/run_e2e_verify.py --config configs/default.yaml --fast
```

Verify the full stack on a synthetic frame (downloads real perception + VLM weights on first run; can take several minutes):

```bash
python scripts/run_e2e_verify.py --config configs/default.yaml
```

Quick wiring check without large downloads:

```bash
python scripts/run_e2e_verify.py --config configs/default.yaml --fast
```

## Perception Backends

The perception tier supports pluggable open-vocabulary detector backends configured in `configs/*.yaml`:

- `grounding_dino` (Transformers-based adapter)
- `yoloe26` (Ultralytics adapter)
- `mock` (explicit-only testing backend; disabled by default)

The embedding backend is also configurable:

- `clip`
- `clip_or_fallback`
- `fallback`

Default behavior is strict Phase-1 path: `grounding_dino` + `clip_or_fallback`, with a real
compact VLM for hazard scoring (`hazard.backend_type: smolvlm`, see `configs/default.yaml`).
Grounding DINO automatically runs on **CPU** when the runtime device is **MPS** (Apple Silicon),
because that checkpoint is unreliable on MPS; override with `perception.detector_device` if needed.
Mock detector fallback is disabled unless `perception.allow_mock_backend=true`.

### Canonical Perception Contract

Perception emits `list[PerceptionDetection]`, where each item contains:

- `track_id`
- `label`
- `confidence`
- `bbox_xyxy`
- `mask`
- `clip_embedding`

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

If Grounding DINO seems stuck on first run (no frames emitted for a long time), it is
usually waiting on Hugging Face weight download/lock resolution. Set `HF_TOKEN`, stop
the run, remove `*.incomplete` files under
`~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base/blobs/`, and retry.
As an alternative, run with YOLOE backend config:

```bash
python scripts/run_perception_smoke.py \
  --config configs/default.yaml \
  --backend-config configs/local_webcam_yoloe.yaml \
  --source 0 \
  --max-frames 30
```

## Hazard Memory (Linear SSM)

The memory module is implemented as a lightweight linear SSM in
`src/risksense_vla/memory/hazard_memory.py` and keeps updates linear per frame:

- recurrence: `x_t = A*x_{t-1} + B*u_t` (implemented via fixed projections + gated update)
- canonical outputs: `MemoryState.objects`, `MemoryState.hoi_embedding`, `MemoryState.state_vector`
- hazard-aware persistence: high-risk objects retain memory longer during missing observations

### Memory APIs

Stateful API (backward compatible with runtime):

```python
from risksense_vla.memory import HazardAwareMemory

memory = HazardAwareMemory()
state = memory.update(
    timestamp=t,
    detections=detections,
    hazards=hazards,  # optional list[float] aligned with detections
)
```

Functional API (explicit previous-state flow):

```python
from risksense_vla.memory import update_hazard_memory

state = update_hazard_memory(
    timestamp=t,
    detections=detections,
    previous_memory_state=prev_state,
    hazards=hazards,  # optional list[float]
    hazard_events=hazard_events,  # optional list[HazardScore]
)
```

Optional compact logging is available through:

- `HazardAwareMemory(log_updates=True)` for module logger output
- `log_callback=callable` in `update(...)` / `update_hazard_memory(...)` for custom sinks

### Memory Validation and Benchmark

Example sequential memory run (with optional per-frame logging):

```bash
python scripts/run_memory_example.py --frames 24 --log
```

Unit tests for dynamic entry/exit and hazard decay:

```bash
python -m pytest tests/unit/test_memory_update.py
```

Benchmark memory update latency/FPS:

```bash
python scripts/eval_memory_fps.py --bench-frames 300 --max-objects 24
```

Benchmark output:

- memory report: `outputs/memory_fps.json`

## Hazard Reasoning (Phase 4)

Hazard reasoning is implemented as a prompt-driven module in
`src/risksense_vla/hazard/` with swappable backends:

- `SmolVlmBackend`: default portable real VLM path (`smolvlm`), using `hazard.vlm_model_id` (e.g. SmolVLM-500M).
- `Phi4MultimodalBackend`: Phi-4 multimodal via `AutoModelForCausalLM` (`phi4_mm`; needs CUDA-class VRAM and `peft`/`backoff`).
- `TinyLocalVLMBackend`: explicit lightweight path (`lightweight_mode=true` only).
- `StubBackend`: deterministic lightweight CI path (`lightweight_mode=true` only).

The main API is exposed by `DistilledHazardReasoner.predict_hazard(...)` (backward-compatible wrapper over `HazardReasoner`):

```python
from risksense_vla.hazard import DistilledHazardReasoner

reasoner = DistilledHazardReasoner(
    backend_type="smolvlm",
    checkpoint_path="artifacts/hazard_reasoner.pt",
    max_tokens=64,
    temperature=0.2,
    lightweight_mode=False,
    vlm_model_id="HuggingFaceTB/SmolVLM-500M-Instruct",
    phi4_model_id="microsoft/Phi-4-multimodal-instruct",
    phi4_precision="int8",
    phi4_estimated_vram_gb=10.0,
    explain=True,
)
out = reasoner.predict_hazard(
    hoi_current=hoi_current,
    hoi_future_embeddings=hoi_future_embeddings,
    memory_state=memory_state,
    frame_bgr=frame_bgr,           # optional
)
```

Returned outputs include:

- `hazard_map` (primary per-track score map)
- `hazard_map_legacy` (`subject:action:object` compatibility map)
- `alerts` / `hazard_alerts`
- per-track natural-language `explanations`
- `global_risk_score`
- `backend_metadata` (model id, precision, estimated VRAM)

### Hazard Config Keys

Configured under `hazard:` in `configs/default.yaml`:

- `backend_type`: `smolvlm` (default), `phi4_mm` (Phi-4 on CUDA), or `tiny`/`stub` with `lightweight_mode=true`
- `vlm_model_id`: Hugging Face repo for `smolvlm` (default `HuggingFaceTB/SmolVLM-500M-Instruct`)
- `max_tokens`: generation cap (default `64`)
- `temperature`: sampling temperature (default `0.2`)
- `lightweight_mode`: enables tiny/stub backends
- `phi4_model_id`, `phi4_precision`, `phi4_estimated_vram_gb` (used for `phi4_mm`)

Phi-4 overlay example:

```bash
python scripts/run_realtime.py --config configs/default.yaml --backend-config configs/phi4_multimodal.yaml
```
- `explain`: include backend explanation text
- `debug_prompt`: include prompt text in logs for debugging
- `reasoner_checkpoint`, `reasoner_fallback_mode`, `alert_threshold`

### Hazard Validation Script

Run synthetic per-frame hazard reasoning with prompt/explanation logging:

```bash
python scripts/run_hazard_reasoner_example.py \
  --backend-type tiny \
  --max-frames 20 \
  --output-jsonl outputs/hazard_reasoner_example.jsonl
```

## Phase-4 Benchmark Gate

Run the strict Phase-4 benchmark over 200 frames and emit:
`outputs/phase4_benchmark.json`

```bash
python scripts/benchmark_phase4.py \
  --config configs/default.yaml \
  --frames 200 \
  --min-fps 10 \
  --require-gpu \
  --output-json outputs/phase4_benchmark.json
```

## Full Pipeline Example

A complete single-frame pipeline example:

```python
from risksense_vla.config import load_config
from risksense_vla.runtime import pick_backend
from risksense_vla.perception import OpenVocabPerception
from risksense_vla.memory import HazardAwareMemory
from risksense_vla.hoi import PredictiveHOIModule
from risksense_vla.hazard import DistilledHazardReasoner
from risksense_vla.attention import SemanticAttentionScheduler
import numpy as np

# Load config
cfg = load_config("configs/default.yaml")
backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))

# Set up modules
perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
memory = HazardAwareMemory()
hoi = PredictiveHOIModule(
    future_horizon_seconds=int(cfg.get("hazard", {}).get("future_horizon_seconds", 3)),
    emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
)
hz = cfg.get("hazard", {})
reasoner = DistilledHazardReasoner(
    alert_threshold=float(hz.get("alert_threshold", 0.65)),
    checkpoint_path=str(hz.get("reasoner_checkpoint", "artifacts/hazard_reasoner.pt")),
    emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
    backend_type=str(hz.get("backend_type", "smolvlm")),
    lightweight_mode=bool(hz.get("lightweight_mode", False)),
    vlm_model_id=str(hz.get("vlm_model_id", "HuggingFaceTB/SmolVLM-500M-Instruct")),
    phi4_model_id=str(hz.get("phi4_model_id", "microsoft/Phi-4-multimodal-instruct")),
)
attention = SemanticAttentionScheduler(
    threshold=float(cfg.get("attention", {}).get("semantic_attention_threshold", 0.6)),
)

# Single-frame pipeline
frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)  # or load from video
timestamp = 0.0

detections = perception.infer(frame_bgr)
mem = memory.update(timestamp=timestamp, detections=detections, hazards=None)
hoi_out = hoi.infer(memory_state=mem, object_detections=detections, timestamp=timestamp)
hazard_out = reasoner.predict_hazard(
    hoi_current=hoi_out.hoi_current,
    hoi_future_embeddings=hoi_out.hoi_future_embeddings,
    memory_state=mem,
    frame_bgr=frame_bgr,
)
schedules = attention.schedule(hazard_map=hazard_out.hazard_map, detections=detections)

# Access results
print("Detections:", len(detections))
print("Hazard map:", hazard_out.hazard_map)
print("Alerts:", hazard_out.alerts)
print("Global risk:", hazard_out.global_risk_score)
```

## Running Ablation Studies

Run ablation studies across memory, HOI, attention, and quantization variants:

**Run all ablations:**
```bash
python scripts/run_ablations.py
```

**Run specific ablations:**
```bash
python scripts/run_ablations.py --ablations baseline naive_memory
```

**Multi-seed runs:**
```bash
python scripts/run_ablations.py --seeds 42 123 456 789 1024
```

**With significance tests (paired t-test, Cohen's d vs baseline):**
```bash
python scripts/run_ablations.py --seeds 42 123 456 789 1024 --compute-significance
```

**Output files** (written to `outputs/ablations/` by default):

- `ablation_results.csv` / `ablation_results_multiseed.csv` — metrics (THC, HAA, RME, mAP, FPS)
- `ablation_results.json` / `ablation_results_multiseed.json` — full JSON results
- `significance_tests.json` — when `--compute-significance` is used
- `plots/` — publication-quality figures

## Reproducing Paper Results

1. **Install with all extras:**
   ```bash
   pip install -e ".[dev,open_vocab,synthetic]"
   ```

2. **Generate synthetic data:**
   ```bash
   python scripts/generate_synthetic_hazards.py --num-scenes 100 --output-dir data/synthetic
   ```

3. **Train all modules:**
   ```bash
   python scripts/train_all.py --config configs/default.yaml --backend-config configs/backend_mps.yaml
   ```

4. **Run ablations with multiple seeds:**
   ```bash
   python scripts/run_ablations.py --seeds 42 123 456 789 1024 --compute-significance
   ```

5. **Generate plots and LaTeX tables** (included in ablation run):
   - Plots: `outputs/ablations/plots/`
   - LaTeX table: generated by `generate_all_plots()` in the same run

## Testing

Install dev dependencies and run tests:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Run stress tests:

```bash
python -m pytest tests/stress/ -v
```

## Model Export

Export trained modules to TorchScript and ONNX:

```bash
python scripts/export_models.py
```

**Options:**

- `--config`, `--backend-config` — config paths
- `--trained-dir` — directory with checkpoints (default: `trained_models`)
- `--out-dir` — output directory (default: `exported_models`)
- `--modules` — modules to export: `perception`, `hoi`, `hazard` (default: all)
- `--convert-qat` — run QAT conversion before export
- `--onnx-opset` — ONNX opset version (default: 17)

**Available formats:** TorchScript (`.pt`), ONNX (`.onnx`)

## Novel Contributions

1. Risk-weighted temporal memory that persists hazardous entities longer.
2. Predictive HOI embeddings with 1-3 second anticipation.
3. Semantic attention scheduling to prioritize high-risk regions.
4. Hazard-aware metrics: THC, HAA, and RME.

