"""Probability calibration metrics and reliability plotting data."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_metrics(y_true, probabilities, *, bins: int = 10) -> dict:
    """Return Brier score, equal-width-bin ECE, and reliability-curve coordinates."""
    y, p = np.asarray(y_true, dtype=int), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0 or bins < 1:
        raise ValueError("Need matching non-empty vectors and bins >= 1")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("labels must be binary and probabilities finite in [0, 1]")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        selected = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if selected.any():
            ece += selected.mean() * abs(float(y[selected].mean() - p[selected].mean()))
    fraction, mean = calibration_curve(y, p, n_bins=bins, strategy="uniform")
    return {"brier": float(brier_score_loss(y, p)), "ece": float(ece),
            "fraction_positive": fraction.tolist(), "mean_probability": mean.tolist(),
            "n_bins_requested": bins, "n": int(y.size)}
