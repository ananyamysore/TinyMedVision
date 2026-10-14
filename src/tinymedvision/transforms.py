"""Deterministic image scaling and seeded, parameterized stress-test corruptions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize(images: torch.Tensor, *, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Normalize a ``(N,C,H,W)`` or ``(C,H,W)`` tensor; inputs are expected in [0, 1]."""
    if std <= 0 or not torch.isfinite(images).all():
        raise ValueError("std must be positive and images must be finite")
    return (images - mean) / std


def corrupt(images: torch.Tensor, kind: str, severity: float, *, seed: int = 0) -> torch.Tensor:
    """Apply a controlled corruption. Severity ranges are validated per corruption."""
    if not torch.isfinite(images).all():
        raise ValueError("images must be finite")
    if not 0 <= severity <= 1:
        raise ValueError("severity must be in [0, 1]")
    batched = images.unsqueeze(0) if images.ndim == 3 else images
    if batched.ndim != 4:
        raise ValueError("images must have shape (C,H,W) or (N,C,H,W)")
    result = batched.clone()
    generator = torch.Generator(device=result.device).manual_seed(seed)
    if kind == "noise":
        result += torch.randn(result.shape, generator=generator, device=result.device,
                              dtype=result.dtype) * (0.25 * severity)
    elif kind == "brightness":
        result += (severity - 0.5) * 0.8
    elif kind == "contrast":
        result = (result - 0.5) * (0.5 + severity) + 0.5
    elif kind == "blur":
        kernel = 1 + 2 * round(2 * severity)
        if kernel > 1:
            result = F.avg_pool2d(result, kernel, stride=1, padding=kernel // 2)
    elif kind == "resolution":
        factor = max(0.1, 1.0 - 0.85 * severity)
        small = F.interpolate(result, scale_factor=factor, mode="bilinear", align_corners=False)
        result = F.interpolate(small, size=batched.shape[-2:], mode="bilinear", align_corners=False)
    elif kind == "compression":
        # Quantization stress approximation, not a JPEG codec implementation.
        levels = max(2, round(256 * (1 - severity) + 2 * severity))
        result = torch.round(result.clamp(0, 1) * (levels - 1)) / (levels - 1)
    else:
        raise ValueError(f"Unknown corruption {kind!r}")
    result = result.clamp(0, 1)
    return result.squeeze(0) if images.ndim == 3 else result
