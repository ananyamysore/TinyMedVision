"""Deterministic toy radiograph-like images for software smoke tests only."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


def generate_fixture(output_dir: str | Path, *, n_per_class: int = 20,
                     size: int = 28, seed: int = 17) -> Path:
    """Write generated grayscale PNGs and a manifest; returns the manifest path.

    Class patterns are deliberately artificial and must never be interpreted as medical evidence.
    Each record gets a unique synthetic source group for leakage-aware split demonstrations.
    """
    if n_per_class < 2 or size < 16:
        raise ValueError("n_per_class must be >=2 and size >=16")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    yy, xx = np.mgrid[:size, :size]
    for label in (0, 1):
        for index in range(n_per_class):
            center = size / 2 + rng.integers(-2, 3, size=2)
            radius = size * (0.2 if label == 0 else 0.29)
            field = 0.22 + 0.15 * np.cos(np.hypot(xx - center[1], yy - center[0]) / 5)
            disk = ((xx - center[1]) ** 2 + (yy - center[0]) ** 2) < radius ** 2
            field[disk] += 0.45
            if label == 1:
                field += 0.12 * np.sin(xx / 2) * np.cos(yy / 3)
            pixels = np.clip(field + rng.normal(0, 0.035, (size, size)), 0, 1)
            filename = f"class{label}_{index:04d}.png"
            Image.fromarray((pixels * 255).astype(np.uint8), mode="L").save(out / filename)
            rows.append({"path": filename, "label": label,
                         "group_id": f"synthetic-{label}-{index:04d}"})
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group_id"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest

