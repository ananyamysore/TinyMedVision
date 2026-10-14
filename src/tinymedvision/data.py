"""Manifest-based image data loading with explicit labels and source identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    """One image record. ``group_id`` identifies a patient, study, or source when available."""

    path: Path
    label: int
    group_id: str | None = None
    split: str | None = None


def read_manifest(path: str | Path, *, root: str | Path | None = None) -> list[Sample]:
    """Read a CSV with required ``path``/``label`` and optional ``group_id``/``split`` columns."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError("Manifest must contain 'path' and 'label' columns")
    if frame.empty:
        raise ValueError("Manifest contains no samples")
    base = Path(root).expanduser().resolve() if root is not None else manifest_path.parent
    samples: list[Sample] = []
    for row in frame.to_dict(orient="records"):
        if pd.isna(row["path"]) or pd.isna(row["label"]):
            raise ValueError("Manifest paths and labels cannot be missing")
        label = int(row["label"])
        if float(row["label"]) != label or label < 0:
            raise ValueError(f"Labels must be non-negative integers; found {row['label']!r}")
        image_path = Path(str(row["path"]))
        if not image_path.is_absolute():
            image_path = base / image_path
        group = row.get("group_id")
        split = row.get("split")
        samples.append(Sample(image_path.resolve(), label,
                              None if pd.isna(group) else str(group),
                              None if pd.isna(split) else str(split)))
    return samples


def validate_samples(samples: list[Sample], *, expected_classes: set[int] | None = None) -> None:
    """Validate labels, image readability, consistent image dimensions and optional class map."""
    if not samples:
        raise ValueError("At least one sample is required")
    labels = {s.label for s in samples}
    if expected_classes is not None and labels != expected_classes:
        raise ValueError(f"Class mapping mismatch: expected {expected_classes}, found {labels}")
    shapes: set[tuple[int, int]] = set()
    for sample in samples:
        if sample.label < 0:
            raise ValueError("Labels must be non-negative")
        if not sample.path.is_file():
            raise FileNotFoundError(f"Image not found: {sample.path}")
        try:
            with Image.open(sample.path) as image:
                shapes.add(image.size)
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Unreadable image: {sample.path}") from exc
    if len(shapes) != 1:
        raise ValueError(f"Images must have matching dimensions; found {sorted(shapes)}")


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load grayscale images from validated manifest records as float tensors in [0, 1]."""

    def __init__(self, samples: list[Sample], *, size: tuple[int, int] | None = None):
        if not samples:
            raise ValueError("At least one sample is required")
        self.samples, self.size = samples, size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("L")
            if self.size is not None:
                image = image.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array[None, ...].copy()), sample.label
