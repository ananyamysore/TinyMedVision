"""Predictive metrics with explicit undefined-class handling and group bootstrap intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    roc_auc_score,
)


def evaluate_predictions(y_true: Sequence[int], probabilities: Sequence[float], *,
                         threshold: float = 0.5) -> dict[str, float | int | None]:
    """Binary metrics; ROC-AUC/AP are ``None`` when only one class is present."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0:
        raise ValueError("y_true and probabilities must be non-empty matching vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("labels must be binary and probabilities finite in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    positives, negatives = tp + fn, tn + fp
    return {
        "n": int(y.size), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if positives else None,
        "sensitivity": float(tp / positives) if positives else None,
        "specificity": float(tn / negatives) if negatives else None,
        "precision": float(precision_score(y, pred, zero_division=0)) if pred.sum() else None,
        "npv": float(tn / (tn + fn)) if (tn + fn) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_auc_ci(y_true: Sequence[int], probabilities: Sequence[float], *,
                     groups: Sequence[str] | None = None, seed: int = 0,
                     n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile CI for ROC-AUC; resamples independent groups when supplied.

    This is descriptive, and assumes independent groups. Returns None if AUC is undefined.
    """
    y, p = np.asarray(y_true), np.asarray(probabilities)
    if y.ndim != 1 or p.shape != y.shape or not y.size or n_bootstrap < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap inputs")
    if np.unique(y).size < 2:
        return None
    if groups is None:
        units = np.arange(y.size)
        membership = [np.array([i]) for i in units]
    else:
        g = np.asarray(groups)
        if g.shape != y.shape or np.any(g == ""):
            raise ValueError("groups must be non-empty and match y_true")
        units, inverse = np.unique(g, return_inverse=True)
        membership = [np.flatnonzero(inverse == i) for i in range(len(units))]
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(units), size=len(units))
        idx = np.concatenate([membership[i] for i in chosen])
        if np.unique(y[idx]).size == 2:
            estimates.append(float(roc_auc_score(y[idx], p[idx])))
    if not estimates:
        return None
    return tuple(float(x) for x in np.quantile(estimates, [alpha / 2, 1 - alpha / 2]))

