"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path


from risksense_vla.config import load_config, merge_dicts, validate_config

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class TestMergeDicts:
    def test_flat_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert merge_dicts(base, override) == {"a": 1, "b": 99}

    def test_nested_merge(self) -> None:
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 9, "z": 3}}
        result = merge_dicts(base, override)
        assert result == {"a": {"x": 1, "y": 9, "z": 3}}

    def test_base_unchanged(self) -> None:
        base = {"a": 1}
        merge_dicts(base, {"a": 2})
        assert base == {"a": 1}


class TestLoadConfig:
    def test_default_yaml_loads(self) -> None:
        cfg = load_config(_CONFIGS_DIR / "default.yaml")
        assert "runtime" in cfg
        assert "perception" in cfg
        assert "hazard" in cfg

    def test_backend_cuda_override(self) -> None:
        cfg = load_config(_CONFIGS_DIR / "default.yaml", _CONFIGS_DIR / "backend_cuda.yaml")
        assert cfg["runtime"]["backend"] == "cuda"

    def test_backend_mps_override(self) -> None:
        cfg = load_config(_CONFIGS_DIR / "default.yaml", _CONFIGS_DIR / "backend_mps.yaml")
        assert cfg["runtime"]["backend"] == "mps"

    def test_ablations_yaml_parses(self) -> None:
        cfg = load_config(_CONFIGS_DIR / "ablations.yaml")
        assert "ablations" in cfg


class TestValidateConfig:
    def test_valid_default_config(self) -> None:
        cfg = load_config(_CONFIGS_DIR / "default.yaml")
        issues = validate_config(cfg)
        assert issues == []

    def test_missing_required_section(self) -> None:
        issues = validate_config({"perception": {}, "hazard": {}})
        assert any("runtime" in i for i in issues)

    def test_invalid_alert_threshold(self) -> None:
        cfg = load_config(_CONFIGS_DIR / "default.yaml")
        cfg["hazard"]["alert_threshold"] = 2.5
        issues = validate_config(cfg)
        assert any("alert_threshold" in i for i in issues)

    def test_invalid_embedding_dim(self) -> None:
        cfg = load_config(_CONFIGS_DIR / "default.yaml")
        cfg["perception"]["embedding_dim"] = -1
        issues = validate_config(cfg)
        assert any("embedding_dim" in i for i in issues)

    def test_empty_config(self) -> None:
        issues = validate_config({})
        assert len(issues) >= len(("runtime", "perception", "hazard"))
