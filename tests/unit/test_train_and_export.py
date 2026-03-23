"""Unit tests for the training, quantization, export, and benchmarking pipeline."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from risksense_vla.train import (
    ModuleTrainer,
    apply_int4_ptq,
    apply_qat,
    benchmark_module,
    convert_qat,
    export_module,
    export_to_onnx,
    export_to_torchscript,
    train_val_split,
)


# ── Fixtures ──────────────────────────────────────────────────────────


class _TinyMLP(nn.Module):
    """Minimal model for testing generic trainer / export plumbing."""

    def __init__(self, in_dim: int = 16, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU(), nn.Linear(8, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TinyCNN(nn.Module):
    """Minimal CNN for perception export tests."""

    def __init__(self, emb_dim: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Linear(8, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x).flatten(1))


class _TinyMultiOutput(nn.Module):
    """Model with tuple output mimicking PredictiveHOINet's signature."""

    def __init__(self, emb_dim: int = 16, num_actions: int = 4, horizon: int = 2):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_actions = num_actions
        self.horizon_seconds = horizon
        self.encoder = nn.Sequential(nn.Linear(2 * emb_dim, 32), nn.ReLU())
        self.current_head = nn.Linear(32, num_actions)
        self.future_action_head = nn.Linear(32, horizon * num_actions)
        self.future_emb_head = nn.Linear(32, horizon * emb_dim)

    def forward(self, obj_emb: torch.Tensor, mem_emb: torch.Tensor):
        z = self.encoder(torch.cat([obj_emb, mem_emb], dim=-1))
        cur = self.current_head(z)
        fut_a = self.future_action_head(z).reshape(-1, self.horizon_seconds, self.num_actions)
        fut_e = self.future_emb_head(z).reshape(-1, self.horizon_seconds, self.emb_dim)
        return cur, fut_a, fut_e


@pytest.fixture
def tiny_dataset():
    x = torch.randn(64, 16)
    y = torch.randint(0, 4, (64,))
    return TensorDataset(x, y)


@pytest.fixture
def tiny_cfg():
    return {"runtime": {"backend": "cpu", "mixed_precision": False}}


@pytest.fixture
def qat_cfg():
    return {
        "runtime": {"backend": "cpu", "mixed_precision": False},
        "qat": {"enabled": True, "fake_quant_backend": "fbgemm", "observer": "histogram"},
    }


@pytest.fixture
def int4_cfg():
    return {
        "runtime": {"backend": "cpu", "mixed_precision": False},
        "int4": {"enabled": True, "calibration_batches": 4},
    }


# ── Train / Val Split ────────────────────────────────────────────────


def test_train_val_split_ratio(tiny_dataset):
    train_ds, val_ds = train_val_split(tiny_dataset, val_ratio=0.2, seed=1)
    assert val_ds is not None
    assert len(train_ds) + len(val_ds) == len(tiny_dataset)


def test_train_val_split_no_val(tiny_dataset):
    train_ds, val_ds = train_val_split(tiny_dataset, val_ratio=0.0)
    assert val_ds is None
    assert len(train_ds) == len(tiny_dataset)


# ── ModuleTrainer ─────────────────────────────────────────────────────


def test_module_trainer_fit_and_checkpoint(tmp_path, tiny_dataset, tiny_cfg):
    model = _TinyMLP(16, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def _loss(pred, target):
        return nn.functional.cross_entropy(pred, target)

    log_path = tmp_path / "log.jsonl"
    trainer = ModuleTrainer(model, opt, _loss, tiny_cfg, log_path=log_path)
    loader = DataLoader(tiny_dataset, batch_size=16, num_workers=0)
    history = trainer.fit(loader, loader, epochs=2)

    assert len(history) == 2
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]

    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt_path, extra={"note": "test"})
    assert ckpt_path.exists()

    model2 = _TinyMLP(16, 4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-2)
    trainer2 = ModuleTrainer(model2, opt2, _loss, tiny_cfg)
    payload = trainer2.load_checkpoint(ckpt_path)
    assert payload["epoch"] == 2
    assert payload["extra"]["note"] == "test"

    # Verify state_dict round-trip
    for key in model.state_dict():
        assert torch.allclose(model.state_dict()[key], model2.state_dict()[key])

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert "epoch" in record and "train_loss" in record


def test_module_trainer_custom_metrics(tmp_path, tiny_dataset, tiny_cfg):
    model = _TinyMLP(16, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def _custom_metrics(m, loader, device):
        return {"custom_acc": 0.85}

    trainer = ModuleTrainer(
        model, opt, nn.CrossEntropyLoss(), tiny_cfg,
        metrics_fn=_custom_metrics,
    )
    loader = DataLoader(tiny_dataset, batch_size=16, num_workers=0)
    history = trainer.fit(loader, loader, epochs=1)
    assert "val_custom_acc" in history[0]
    assert abs(history[0]["val_custom_acc"] - 0.85) < 1e-9


# ── QAT ───────────────────────────────────────────────────────────────


def test_qat_prepare_and_convert_mlp(qat_cfg):
    model = _TinyMLP(16, 4)
    prepared = apply_qat(model, qat_cfg)
    x = torch.randn(4, 16)
    out = prepared(x)
    assert out.shape == (4, 4)

    converted = convert_qat(prepared)  # gracefully returns model on unsupported platforms
    out2 = converted(x)
    assert out2.shape == (4, 4)


def test_qat_prepare_and_convert_cnn(qat_cfg):
    model = _TinyCNN(16)
    prepared = apply_qat(model, qat_cfg)
    x = torch.randn(2, 3, 32, 32)
    out = prepared(x)
    assert out.shape == (2, 16)


def test_qat_disabled_returns_unchanged(tiny_cfg):
    model = _TinyMLP(16, 4)
    result = apply_qat(model, tiny_cfg)
    assert result is model


# ── INT4 PTQ ──────────────────────────────────────────────────────────


def test_int4_ptq_runs(int4_cfg, tiny_dataset):
    model = _TinyMLP(16, 4)
    cal_loader = DataLoader(tiny_dataset, batch_size=8, num_workers=0)
    result = apply_int4_ptq(model, cal_loader, int4_cfg, device="cpu")
    x = torch.randn(4, 16)
    out = result(x)
    assert out.shape == (4, 4)


def test_int4_ptq_disabled(tiny_cfg, tiny_dataset):
    model = _TinyMLP(16, 4)
    cal_loader = DataLoader(tiny_dataset, batch_size=8, num_workers=0)
    result = apply_int4_ptq(model, cal_loader, tiny_cfg, device="cpu")
    assert result is model


# ── TorchScript Export ────────────────────────────────────────────────


def test_export_torchscript_mlp(tmp_path):
    model = _TinyMLP(16, 4)
    dummy = torch.randn(2, 16)
    ts_path = export_to_torchscript(model, dummy, tmp_path / "model.ts")
    assert ts_path.exists()

    loaded = torch.jit.load(str(ts_path))
    out_orig = model(dummy)
    out_loaded = loaded(dummy)
    assert torch.allclose(out_orig, out_loaded, atol=1e-5)


def test_export_torchscript_cnn(tmp_path):
    model = _TinyCNN(16)
    dummy = torch.randn(1, 3, 32, 32)
    ts_path = export_to_torchscript(model, dummy, tmp_path / "perception.ts")
    assert ts_path.exists()

    loaded = torch.jit.load(str(ts_path))
    out = loaded(dummy)
    assert out.shape == (1, 16)


def test_export_torchscript_multi_output(tmp_path):
    model = _TinyMultiOutput(emb_dim=16, num_actions=4, horizon=2)
    dummy = (torch.randn(1, 16), torch.randn(1, 16))
    ts_path = export_to_torchscript(model, dummy, tmp_path / "hoi.ts")
    assert ts_path.exists()

    loaded = torch.jit.load(str(ts_path))
    cur, fut_a, fut_e = loaded(*dummy)
    assert cur.shape == (1, 4)
    assert fut_a.shape == (1, 2, 4)
    assert fut_e.shape == (1, 2, 16)


# ── ONNX Export ───────────────────────────────────────────────────────


def test_export_onnx_mlp(tmp_path):
    model = _TinyMLP(16, 4)
    dummy = torch.randn(2, 16)
    onnx_path = export_to_onnx(
        model, dummy, tmp_path / "model.onnx",
        input_names=["features"], output_names=["logits"],
    )
    assert onnx_path.exists()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    inputs = {sess.get_inputs()[0].name: dummy.numpy()}
    outputs = sess.run(None, inputs)
    assert outputs[0].shape == (2, 4)


def test_export_onnx_cnn(tmp_path):
    model = _TinyCNN(16)
    dummy = torch.randn(1, 3, 32, 32)
    onnx_path = export_to_onnx(
        model, dummy, tmp_path / "perception.onnx",
        input_names=["image"], output_names=["embedding"],
    )
    assert onnx_path.exists()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    outputs = sess.run(None, {"image": dummy.numpy()})
    assert outputs[0].shape == (1, 16)


def test_export_onnx_multi_output(tmp_path):
    model = _TinyMultiOutput(emb_dim=16, num_actions=4, horizon=2)
    dummy = (torch.randn(1, 16), torch.randn(1, 16))
    onnx_path = export_to_onnx(
        model, dummy, tmp_path / "hoi.onnx",
        input_names=["object_emb", "memory_emb"],
        output_names=["current", "future_action", "future_emb"],
    )
    assert onnx_path.exists()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    inputs = {
        "object_emb": dummy[0].numpy(),
        "memory_emb": dummy[1].numpy(),
    }
    outputs = sess.run(None, inputs)
    assert len(outputs) == 3
    assert outputs[0].shape == (1, 4)


# ── export_module (high-level) ────────────────────────────────────────


def test_export_module_perception(tmp_path):
    model = _TinyCNN(16)
    cfg = {"optimization": {"export_formats": ["torchscript", "onnx"]}}

    from risksense_vla.train.export import _DUMMY_SPECS
    _DUMMY_SPECS["test_perception"] = {
        "dummy_fn": lambda: torch.randn(1, 3, 32, 32),
        "input_names": ["image"],
        "output_names": ["embedding"],
    }
    results = export_module(model, "test_perception", cfg, tmp_path)
    assert "torchscript" in results
    assert "onnx" in results
    assert results["torchscript"].exists()
    assert results["onnx"].exists()
    del _DUMMY_SPECS["test_perception"]


# ── Benchmark ─────────────────────────────────────────────────────────


def test_benchmark_module_returns_metrics():
    model = _TinyMLP(16, 4)
    dummy = torch.randn(1, 16)
    result = benchmark_module(model, dummy, warmup=5, iterations=10, device="cpu")

    for key in ("avg_latency_ms", "p50_ms", "p95_ms", "max_latency_ms", "fps", "peak_memory_mb"):
        assert key in result, f"Missing key: {key}"
    assert result["fps"] > 0
    assert result["avg_latency_ms"] > 0


def test_benchmark_module_multi_input():
    model = _TinyMultiOutput(emb_dim=16, num_actions=4, horizon=2)
    dummy = (torch.randn(1, 16), torch.randn(1, 16))
    result = benchmark_module(model, dummy, warmup=5, iterations=10, device="cpu")
    assert result["fps"] > 0


# ── Quantization Shape Validation ─────────────────────────────────────


def test_qat_preserves_output_shapes():
    model = _TinyMultiOutput(emb_dim=16, num_actions=4, horizon=2)
    dummy = (torch.randn(1, 16), torch.randn(1, 16))
    fp32_out = model(*dummy)

    cfg = {"qat": {"enabled": True, "fake_quant_backend": "fbgemm", "observer": "histogram"}}
    prepared = apply_qat(model, cfg)
    qat_out = prepared(*dummy)

    assert len(fp32_out) == len(qat_out)
    for fp, qat in zip(fp32_out, qat_out):
        assert fp.shape == qat.shape, f"Shape mismatch: {fp.shape} vs {qat.shape}"

    converted = convert_qat(prepared)
    conv_out = converted(*dummy)
    assert len(fp32_out) == len(conv_out)
    for fp, co in zip(fp32_out, conv_out):
        assert fp.shape == co.shape, f"Shape mismatch: {fp.shape} vs {co.shape}"


def test_int4_ptq_preserves_output_shapes():
    model = _TinyMLP(16, 4)
    dummy = torch.randn(4, 16)
    fp32_out = model(dummy)

    cal_data = TensorDataset(torch.randn(32, 16), torch.randint(0, 4, (32,)))
    cal_loader = DataLoader(cal_data, batch_size=8, num_workers=0)
    cfg = {"int4": {"enabled": True, "calibration_batches": 4}}
    quantized = apply_int4_ptq(model, cal_loader, cfg, device="cpu")
    q_out = quantized(dummy)

    assert fp32_out.shape == q_out.shape


def test_exported_torchscript_roundtrip(tmp_path):
    """Verify TorchScript export produces identical outputs on reload."""
    model = _TinyMLP(16, 4)
    model.eval()
    dummy = torch.randn(2, 16)
    with torch.no_grad():
        expected = model(dummy)

    ts_path = export_to_torchscript(model, dummy, tmp_path / "roundtrip.ts")
    loaded = torch.jit.load(str(ts_path))
    loaded.eval()
    with torch.no_grad():
        actual = loaded(dummy)

    assert torch.allclose(expected, actual, atol=1e-5)


def test_exported_onnx_roundtrip(tmp_path):
    """Verify ONNX export produces numerically close outputs."""
    model = _TinyMLP(16, 4)
    model.eval()
    dummy = torch.randn(2, 16)
    with torch.no_grad():
        expected = model(dummy).numpy()

    onnx_path = export_to_onnx(
        model, dummy, tmp_path / "roundtrip.onnx",
        input_names=["x"], output_names=["y"],
    )

    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(str(onnx_path))
    actual = sess.run(None, {"x": dummy.numpy()})[0]
    assert np.allclose(expected, actual, atol=1e-4)
