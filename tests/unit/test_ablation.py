"""Tests for the ablation study framework."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
import torch

from risksense_vla.eval.ablation import (
    ABLATION_REGISTRY,
    AblationConfig,
    AblationRunner,
    NaiveMemory,
    UniformAttentionScheduler,
    _generate_synthetic_sequence,
    build_pipeline,
    results_to_csv,
    seed_everything,
)
from risksense_vla.types import HazardScore, MemoryState, PerceptionDetection


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


class TestAblationConfig:
    def test_valid_baseline(self) -> None:
        cfg = AblationConfig(name="test", memory_mode="hazard_aware", hoi_mode="predictive")
        assert cfg.memory_mode == "hazard_aware"
        assert cfg.seed == 42

    def test_invalid_memory_mode(self) -> None:
        with pytest.raises(ValueError, match="memory_mode"):
            AblationConfig(name="bad", memory_mode="transformer")

    def test_invalid_hoi_mode(self) -> None:
        with pytest.raises(ValueError, match="hoi_mode"):
            AblationConfig(name="bad", hoi_mode="rnn")

    def test_invalid_attention_mode(self) -> None:
        with pytest.raises(ValueError, match="attention_mode"):
            AblationConfig(name="bad", attention_mode="cross")

    def test_invalid_quant_mode(self) -> None:
        with pytest.raises(ValueError, match="quant_mode"):
            AblationConfig(name="bad", quant_mode="int2")


class TestAblationRegistry:
    def test_registry_has_baseline(self) -> None:
        assert "baseline" in ABLATION_REGISTRY

    def test_registry_has_all_ablations(self) -> None:
        expected = {"baseline", "naive_memory", "frame_only_hoi", "uniform_attention", "int8_qat", "int4_ptq", "int8_masked"}
        assert expected.issubset(set(ABLATION_REGISTRY.keys()))

    def test_each_config_is_valid(self) -> None:
        for name, cfg in ABLATION_REGISTRY.items():
            assert cfg.name == name
            assert isinstance(cfg, AblationConfig)

    def test_baseline_is_full_default(self) -> None:
        b = ABLATION_REGISTRY["baseline"]
        assert b.memory_mode == "hazard_aware"
        assert b.hoi_mode == "predictive"
        assert b.attention_mode == "semantic"
        assert b.quant_mode == "fp32"

    @pytest.mark.parametrize("name,field,expected", [
        ("naive_memory", "memory_mode", "naive"),
        ("frame_only_hoi", "hoi_mode", "frame_only"),
        ("uniform_attention", "attention_mode", "uniform"),
        ("int8_qat", "quant_mode", "int8"),
        ("int4_ptq", "quant_mode", "int4_ptq"),
        ("int8_masked", "quant_mode", "int8_masked"),
    ])
    def test_single_axis_change(self, name: str, field: str, expected: str) -> None:
        cfg = ABLATION_REGISTRY[name]
        assert getattr(cfg, field) == expected


# ---------------------------------------------------------------------------
# NaiveMemory
# ---------------------------------------------------------------------------


def _make_detection(track_id: str = "obj_0", label: str = "knife") -> PerceptionDetection:
    return PerceptionDetection(
        track_id=track_id,
        label=label,
        confidence=0.9,
        bbox_xyxy=(10, 10, 50, 50),
        mask=torch.zeros((1, 1), dtype=torch.float32),
        clip_embedding=torch.randn(256, dtype=torch.float32),
    )


class TestNaiveMemory:
    def test_update_returns_memory_state(self) -> None:
        mem = NaiveMemory(emb_dim=256)
        dets = [_make_detection("obj_0"), _make_detection("obj_1")]
        state = mem.update(timestamp=0.0, detections=dets)
        assert isinstance(state, MemoryState)
        assert len(state.objects) == 2
        assert state.hoi_embedding.shape == (1, 256)
        assert state.state_vector.shape == (1, 512)

    def test_persistence_decays_without_observation(self) -> None:
        mem = NaiveMemory(emb_dim=256)
        dets = [_make_detection("obj_0")]
        s1 = mem.update(timestamp=0.0, detections=dets)
        initial_p = s1.objects[0].persistence

        s2 = mem.update(timestamp=1.0, detections=[], previous_memory_state=s1)
        if s2.objects:
            assert s2.objects[0].persistence < initial_p

    def test_no_hazard_weighting(self) -> None:
        mem = NaiveMemory(emb_dim=256)
        dets = [_make_detection("obj_0")]
        state = mem.update(
            timestamp=0.0,
            detections=dets,
            hazards=[0.95],
            hazard_events=[
                HazardScore(subject="human", action="cut", object="knife", score=0.95, severity="high", explanation="")
            ],
        )
        assert state.objects[0].hazard_weight == 0.0

    def test_empty_detections(self) -> None:
        mem = NaiveMemory(emb_dim=256)
        state = mem.update(timestamp=0.0, detections=[])
        assert isinstance(state, MemoryState)
        assert len(state.objects) == 0


# ---------------------------------------------------------------------------
# UniformAttentionScheduler
# ---------------------------------------------------------------------------


class TestUniformAttentionScheduler:
    def test_uniform_allocation(self) -> None:
        sched = UniformAttentionScheduler()
        dets = [_make_detection("a"), _make_detection("b"), _make_detection("c")]
        hazards = [
            HazardScore(subject="human", action="cut", object="a", score=0.9, severity="high", explanation=""),
        ]
        alloc = sched.allocation(dets, hazards)
        assert len(alloc) == 3
        assert all(v == 1.0 for v in alloc.values())

    def test_empty_detections(self) -> None:
        sched = UniformAttentionScheduler()
        assert sched.allocation([], []) == {}


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------


class TestSeedEverything:
    def test_deterministic_torch(self) -> None:
        seed_everything(123)
        a = torch.randn(10)
        seed_everything(123)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_deterministic_numpy(self) -> None:
        import numpy as np

        seed_everything(456)
        a = np.random.rand(10)
        seed_everything(456)
        b = np.random.rand(10)
        assert (a == b).all()

    def test_different_seeds_differ(self) -> None:
        seed_everything(1)
        a = torch.randn(10)
        seed_everything(2)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    @pytest.fixture
    def base_cfg(self) -> dict:
        return {
            "perception": {"embedding_dim": 256},
            "hazard": {"future_horizon_seconds": 3},
            "attention": {"semantic_attention_threshold": 0.6, "low_risk_scale": 0.5, "high_risk_scale": 1.0},
            "optimization": {"pruning_ratio": 0.2},
        }

    def test_baseline_pipeline(self, base_cfg: dict) -> None:
        from risksense_vla.attention.semantic_scheduler import SemanticAttentionScheduler
        from risksense_vla.hoi import PredictiveHOIModule
        from risksense_vla.memory.hazard_memory import HazardAwareMemory

        pipe = build_pipeline(ABLATION_REGISTRY["baseline"], base_cfg)
        assert isinstance(pipe.memory, HazardAwareMemory)
        assert isinstance(pipe.hoi, PredictiveHOIModule)
        assert isinstance(pipe.attention, SemanticAttentionScheduler)

    def test_naive_memory_pipeline(self, base_cfg: dict) -> None:
        pipe = build_pipeline(ABLATION_REGISTRY["naive_memory"], base_cfg)
        assert isinstance(pipe.memory, NaiveMemory)

    def test_frame_only_hoi_pipeline(self, base_cfg: dict) -> None:
        from risksense_vla.hoi import ProtoHOIPredictor

        pipe = build_pipeline(ABLATION_REGISTRY["frame_only_hoi"], base_cfg)
        assert isinstance(pipe.hoi, ProtoHOIPredictor)

    def test_uniform_attention_pipeline(self, base_cfg: dict) -> None:
        pipe = build_pipeline(ABLATION_REGISTRY["uniform_attention"], base_cfg)
        assert isinstance(pipe.attention, UniformAttentionScheduler)

    def test_pipeline_respects_use_hazard_weighting_toggle(self, base_cfg: dict) -> None:
        from risksense_vla.memory.hazard_memory import HazardAwareMemory

        cfg = dict(base_cfg)
        cfg["memory"] = {"use_hazard_weighting": False}
        pipe = build_pipeline(ABLATION_REGISTRY["baseline"], cfg)
        assert isinstance(pipe.memory, HazardAwareMemory)
        assert pipe.memory.use_hazard_weighting is False


# ---------------------------------------------------------------------------
# Synthetic sequence generation
# ---------------------------------------------------------------------------


class TestSyntheticSequence:
    def test_generates_correct_length(self) -> None:
        seq = _generate_synthetic_sequence(num_frames=50, seed=42)
        assert len(seq) == 50

    def test_records_have_required_fields(self) -> None:
        seq = _generate_synthetic_sequence(num_frames=10, seed=42)
        for rec in seq:
            assert "frame_id" in rec
            assert "detections" in rec
            assert "hois" in rec
            assert "hazards" in rec
            assert "latency_ms" in rec

    def test_deterministic(self) -> None:
        a = _generate_synthetic_sequence(num_frames=20, seed=99)
        b = _generate_synthetic_sequence(num_frames=20, seed=99)
        assert a == b


# ---------------------------------------------------------------------------
# AblationRunner (lightweight integration)
# ---------------------------------------------------------------------------


class TestAblationRunner:
    @pytest.fixture
    def runner(self) -> AblationRunner:
        cfg = {
            "perception": {"embedding_dim": 256},
            "hazard": {"future_horizon_seconds": 3},
            "attention": {"semantic_attention_threshold": 0.6, "low_risk_scale": 0.5, "high_risk_scale": 1.0},
            "optimization": {"pruning_ratio": 0.2},
        }
        return AblationRunner(cfg=cfg, dataset_dir="nonexistent_dir", warmup=2, iterations=5, device="cpu")

    def test_run_single_baseline(self, runner: AblationRunner) -> None:
        result = runner.run_single(ABLATION_REGISTRY["baseline"])
        assert result.config_name == "baseline"
        assert result.thc >= 0.0
        assert result.fps >= 0.0

    def test_run_single_naive_memory(self, runner: AblationRunner) -> None:
        result = runner.run_single(ABLATION_REGISTRY["naive_memory"])
        assert result.config_name == "naive_memory"
        assert result.config.memory_mode == "naive"

    def test_run_all_returns_list(self, runner: AblationRunner) -> None:
        configs = [ABLATION_REGISTRY["baseline"], ABLATION_REGISTRY["uniform_attention"]]
        results = runner.run_all(configs)
        assert len(results) == 2
        assert results[0].config_name == "baseline"
        assert results[1].config_name == "uniform_attention"


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class TestCSVOutput:
    def test_csv_has_correct_columns(self) -> None:
        from risksense_vla.eval.ablation import AblationResult, _CSV_COLUMNS

        results = [
            AblationResult(
                config_name="baseline",
                config=ABLATION_REGISTRY["baseline"],
                thc=0.75, haa=0.60, rme=0.30, detection_map=0.45, fps=28.5,
                latency_ms=35.0, peak_memory_mb=120.0, seed=42,
            ),
            AblationResult(
                config_name="naive_memory",
                config=ABLATION_REGISTRY["naive_memory"],
                thc=0.65, haa=0.55, rme=0.25, detection_map=0.42, fps=30.0,
                latency_ms=33.0, peak_memory_mb=115.0, seed=42,
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmp_path = f.name
        try:
            results_to_csv(results, tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert set(rows[0].keys()) == set(_CSV_COLUMNS)
            assert rows[0]["ablation"] == "baseline"
            assert rows[0]["method_name"] == "HW-SSM (Proposed)"
            assert float(rows[0]["THC"]) == 0.75
            assert float(rows[1]["delta_THC_pct"]) != 0.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_csv_delta_baseline_is_zero(self) -> None:
        from risksense_vla.eval.ablation import AblationResult

        results = [
            AblationResult(
                config_name="baseline",
                config=ABLATION_REGISTRY["baseline"],
                thc=0.75, haa=0.60, rme=0.30, detection_map=0.45, fps=28.5,
                latency_ms=35.0, peak_memory_mb=120.0, seed=42,
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmp_path = f.name
        try:
            results_to_csv(results, tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert float(rows[0]["delta_THC_pct"]) == 0.0
            assert float(rows[0]["delta_FPS_pct"]) == 0.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------


class TestYAMLConfigLoading:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        from risksense_vla.eval.ablation import load_ablation_configs_from_yaml

        yaml_content = """
seed: 99
ablations:
  test_baseline:
    memory_mode: hazard_aware
    hoi_mode: predictive
    attention_mode: semantic
    quant_mode: fp32
  test_naive:
    memory_mode: naive
"""
        cfg_file = tmp_path / "test_ablations.yaml"
        cfg_file.write_text(yaml_content, encoding="utf-8")
        configs = load_ablation_configs_from_yaml(cfg_file)
        assert "test_baseline" in configs
        assert "test_naive" in configs
        assert configs["test_naive"].memory_mode == "naive"
        assert configs["test_naive"].seed == 99
        assert configs["test_baseline"].hoi_mode == "predictive"
