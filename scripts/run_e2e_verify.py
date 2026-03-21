#!/usr/bin/env python3
"""Run one frame through perception → memory → HOI → hazard (optional fast stub path)."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time

import numpy as np

from risksense_vla.config import load_config
from risksense_vla.eval.ablation import NaiveMemory
from risksense_vla.experimental import apply_occlusion, get_bool, resolve_mode, seed_everything
from risksense_vla.hazard import DistilledHazardReasoner
from risksense_vla.hoi import PredictiveHOIModule, ProtoHOIPredictor
from risksense_vla.memory import HazardAwareMemory
from risksense_vla.perception import OpenVocabPerception
from risksense_vla.runtime import pick_backend

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default=None)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Mock detector, fallback embedder, stub hazard (no HF weight downloads).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    cfg = load_config(args.config, args.backend_config)
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    seed_everything(seed)
    rng = random.Random(seed)
    if args.fast:
        p = dict(cfg.get("perception", {}))
        p.update(
            {
                "detector_backend": "mock",
                "allow_mock_backend": True,
                "embedder_backend": "fallback",
            }
        )
        h = dict(cfg.get("hazard", {}))
        h.update({"backend_type": "stub", "lightweight_mode": True})
        cfg = {**cfg, "perception": p, "hazard": h}
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    _LOG.info("backend=%s device=%s", backend.name, backend.device)
    memory_mode = resolve_mode(cfg, "memory_mode", "hazard_aware")
    hoi_mode = resolve_mode(cfg, "hoi_mode", "predictive")
    use_hazard_weighting = get_bool(cfg, "memory", "use_hazard_weighting", True)
    use_prediction = get_bool(cfg, "hoi", "use_prediction", True)
    use_vlm = get_bool(cfg, "hazard", "use_vlm", True)
    occlusion_prob = float(cfg.get("evaluation", {}).get("occlusion_prob", 0.0))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[120:360, 160:480] = (40, 180, 40)

    t0 = time.perf_counter()
    perception = OpenVocabPerception.from_config(cfg=cfg, device=backend.device)
    detections = perception.infer(frame)
    detections, occlusion_events = apply_occlusion(detections, occlusion_prob=occlusion_prob, rng=rng)
    t1 = time.perf_counter()
    _LOG.info("perception: %d detections in %.2fs", len(detections), t1 - t0)

    if memory_mode == "naive":
        memory = NaiveMemory(emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)))
    else:
        memory = HazardAwareMemory(
            emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
            alpha=0.0 if not use_hazard_weighting else 0.14,
            beta=0.0 if not use_hazard_weighting else 1.4,
            use_hazard_weighting=use_hazard_weighting,
        )
    mem = memory.update(timestamp=0.0, detections=detections, hazards=None)
    if hoi_mode == "frame_only" or not use_prediction:
        hoi: PredictiveHOIModule | ProtoHOIPredictor = ProtoHOIPredictor(
            future_horizon_seconds=int(cfg.get("hazard", {}).get("future_horizon_seconds", 3)),
            emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
        )
        hoi_triplets = hoi.predict(timestamp=0.0, detections=detections, memory=mem)
        hoi_current = [h for h in hoi_triplets if not h.predicted]
        hoi_future_embeddings = None
    else:
        hoi = PredictiveHOIModule(
            future_horizon_seconds=int(cfg.get("hazard", {}).get("future_horizon_seconds", 3)),
            emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
        )
        hoi_out = hoi.infer(memory_state=mem, object_detections=detections, timestamp=0.0)
        hoi_current = hoi_out.hoi_current
        hoi_future_embeddings = hoi_out.hoi_future_embeddings

    hz = cfg.get("hazard", {})
    backend_type = str(hz.get("backend_type", "smolvlm"))
    lightweight_mode = bool(hz.get("lightweight_mode", False))
    if not use_vlm:
        backend_type = "stub"
        lightweight_mode = True
    reasoner = DistilledHazardReasoner(
        alert_threshold=float(hz.get("alert_threshold", 0.65)),
        checkpoint_path=str(hz.get("reasoner_checkpoint", "artifacts/hazard_reasoner.pt")),
        fallback_mode=str(hz.get("reasoner_fallback_mode", "blend")),
        emb_dim=int(cfg.get("perception", {}).get("embedding_dim", 256)),
        backend_type=backend_type,
        max_tokens=int(hz.get("max_tokens", 64)),
        temperature=float(hz.get("temperature", 0.2)),
        quantized=bool(hz.get("quantized", False)),
        lightweight_mode=lightweight_mode,
        phi4_model_id=str(hz.get("phi4_model_id", "microsoft/Phi-4-multimodal-instruct")),
        phi4_precision=str(hz.get("phi4_precision", "int8")),
        phi4_estimated_vram_gb=float(hz.get("phi4_estimated_vram_gb", 10.0)),
        vlm_model_id=str(hz.get("vlm_model_id", "HuggingFaceTB/SmolVLM-500M-Instruct")),
        explain=bool(hz.get("explain", True)),
        debug_prompt=bool(hz.get("debug_prompt", False)),
    )
    t2 = time.perf_counter()
    hazard_out = reasoner.predict_hazard(
        hoi_current=hoi_current,
        hoi_future_embeddings=hoi_future_embeddings,
        memory_state=mem,
        frame_bgr=frame,
    )
    t3 = time.perf_counter()
    _LOG.info(
        "hazard: backend=%s global_risk=%.3f hois=%d in %.2fs",
        hazard_out.backend,
        hazard_out.global_risk_score,
        len(hazard_out.hazards),
        t3 - t2,
    )
    if not hazard_out.hazards:
        _LOG.warning("No hazard scores (empty HOI list).")
        return 1
    if not use_prediction:
        assert hoi_mode == "frame_only" or not use_prediction
    if not use_vlm:
        assert hazard_out.backend == "stub"
    if not use_hazard_weighting and isinstance(memory, HazardAwareMemory):
        assert abs(memory.alpha) < 1e-9 and abs(memory.beta) < 1e-9 and not memory.use_hazard_weighting
    if occlusion_prob > 0.0:
        _LOG.info("occlusion events=%d", len(occlusion_events))
    for h in hazard_out.hazards[:3]:
        _LOG.info("  score=%.3f %s %s %s", h.score, h.subject, h.action, h.object)
    _LOG.info("e2e_ok total_wall=%.2fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
