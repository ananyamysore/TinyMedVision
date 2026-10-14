"""Same-sample evaluation under deterministic image-quality stress tests."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import evaluate_predictions
from .transforms import corrupt


def evaluate_corruptions(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, *,
                         kinds: tuple[str, ...] = ("noise", "blur", "brightness", "contrast",
                                                   "resolution", "compression"),
                         severities: tuple[float, ...] = (0.25, 0.5, 0.75),
                         batch_size: int = 64, seed: int = 0,
                         device: str = "cpu") -> list[dict]:
    """Evaluate each corruption on the exact same examples and return condition records."""
    if images.shape[0] != labels.numel() or images.shape[0] == 0:
        raise ValueError("images and labels must contain the same non-zero sample count")
    results = []
    for kind in kinds:
        for severity in severities:
            shifted = corrupt(images, kind, severity, seed=seed)
            probs: list[float] = []
            model.eval()
            with torch.inference_mode():
                for (batch,) in DataLoader(TensorDataset(shifted), batch_size=batch_size):
                    logits = model(batch.to(device))
                    probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            results.append({"corruption": kind, "severity": severity, "seed": seed,
                            **evaluate_predictions(labels.cpu().numpy(), probs)})
    return results
