"""Measured model size, parameter counts, and repeatable inference timing."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import torch
from torch import nn


def benchmark_model(model: nn.Module, *, input_shape: tuple[int, int, int] = (1, 28, 28),
                    batch_size: int = 1, warmup: int = 10, iterations: int = 100,
                    device: str = "cpu", checkpoint: str | Path | None = None) -> dict:
    """Measure inference latency with documented workload and runtime metadata."""
    if min(input_shape) < 1 or batch_size < 1 or warmup < 0 or iterations < 1:
        raise ValueError("Invalid benchmark dimensions or iteration counts")
    target = torch.device(device)
    model = model.to(target).eval()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    inputs = torch.zeros((batch_size, *input_shape), device=target)
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(inputs)
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            timings.append(time.perf_counter() - start)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = None
    if checkpoint is not None:
        size_bytes = Path(checkpoint).stat().st_size
    return {"parameters": parameter_count, "trainable_parameters": trainable,
            "serialized_bytes": size_bytes, "latency_median_ms": 1000 * float(torch.tensor(timings).median()),
            "latency_p95_ms": 1000 * float(torch.quantile(torch.tensor(timings), 0.95)),
            "throughput_images_per_second": batch_size / (sum(timings) / len(timings)),
            "batch_size": batch_size, "input_shape": list(input_shape), "warmup": warmup,
            "iterations": iterations, "device": str(target), "hardware": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None}
