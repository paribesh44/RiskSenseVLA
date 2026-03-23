"""Per-module inference benchmarking: latency, FPS, peak memory."""

from __future__ import annotations

import resource
import statistics
import time

import torch
import torch.nn as nn


def _forward(model: nn.Module, inp: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
    """Single forward pass that handles both single-tensor and tuple inputs."""
    if isinstance(inp, tuple):
        model(*inp)
    else:
        model(inp)


def _move_to_device(
    inp: torch.Tensor | tuple[torch.Tensor, ...],
    dev: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(inp, tuple):
        return tuple(t.to(dev) for t in inp)
    return inp.to(dev)


def _measure_cuda(
    model: nn.Module,
    dummy_input: torch.Tensor | tuple[torch.Tensor, ...],
    dev: torch.device,
    iterations: int,
) -> list[float]:
    latencies: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _forward(model, dummy_input)
        end.record()
        torch.cuda.synchronize(dev)
        latencies.append(start.elapsed_time(end))
    return latencies


def _measure_cpu(
    model: nn.Module,
    dummy_input: torch.Tensor | tuple[torch.Tensor, ...],
    iterations: int,
) -> list[float]:
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _forward(model, dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def benchmark_module(
    model: nn.Module,
    dummy_input: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    warmup: int = 50,
    iterations: int = 200,
    device: str = "cpu",
) -> dict[str, float]:
    """Run *model* repeatedly and return latency / throughput statistics.

    Returns a dict with ``avg_latency_ms``, ``p50_ms``, ``p95_ms``,
    ``max_latency_ms``, ``fps``, and ``peak_memory_mb``.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    dummy_input = _move_to_device(dummy_input, dev)
    use_cuda = dev.type == "cuda"

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.inference_mode():
        for _ in range(warmup):
            _forward(model, dummy_input)

        if use_cuda:
            torch.cuda.synchronize(dev)

        mem_before = _current_memory_mb(dev)
        latencies = (
            _measure_cuda(model, dummy_input, dev, iterations)
            if use_cuda
            else _measure_cpu(model, dummy_input, iterations)
        )

    peak_mem = _peak_memory_mb(dev, mem_before)
    avg = statistics.mean(latencies)
    latencies_sorted = sorted(latencies)
    p50 = _percentile(latencies_sorted, 50)
    p95 = _percentile(latencies_sorted, 95)
    return {
        "avg_latency_ms": round(avg, 4),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "max_latency_ms": round(max(latencies), 4),
        "fps": round(1000.0 / avg, 2) if avg > 0 else 0.0,
        "peak_memory_mb": round(peak_mem, 2),
    }


def _percentile(sorted_values: list[float], pct: int) -> float:
    idx = int(len(sorted_values) * pct / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def _current_memory_mb(dev: torch.device) -> float:
    if dev.type == "cuda":
        return torch.cuda.memory_allocated(dev) / (1024 * 1024)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _peak_memory_mb(dev: torch.device, baseline_mb: float) -> float:
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
    current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    return max(0.0, current - baseline_mb)
