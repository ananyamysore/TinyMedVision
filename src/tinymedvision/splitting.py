"""Leakage-aware partition validation and deterministic group splitting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data import Sample


def validate_split_groups(samples: Sequence[Sample]) -> None:
    """Raise when a non-missing source group appears in more than one declared split."""
    group_splits: dict[str, set[str]] = {}
    for sample in samples:
        if sample.group_id and sample.split:
            group_splits.setdefault(sample.group_id, set()).add(sample.split)
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        raise ValueError(f"Source groups cross split boundaries: {leaking[:5]}")


def split_indices(labels: Sequence[int], *, test_size: float = 0.2,
                  groups: Sequence[str] | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create a seeded holdout; groups, when supplied, stay intact across partitions."""
    y = np.asarray(labels)
    if y.ndim != 1 or y.size < 2 or not 0 < test_size < 1:
        raise ValueError("Need >=2 labels and test_size strictly between 0 and 1")
    indices = np.arange(y.size)
    if groups is None:
        return tuple(np.asarray(part) for part in train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y))  # type: ignore[return-value]
    group_array = np.asarray(groups)
    if group_array.shape != y.shape or np.any(group_array == ""):
        raise ValueError("groups must be non-empty and match labels")
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                         random_state=seed).split(indices, y, group_array))
    return train, test
