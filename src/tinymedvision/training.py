"""Seeded CPU/GPU training loops and portable state-dict checkpointing."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds (determinism remains backend-dependent)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, *,
                epochs: int = 3, lr: float = 1e-3, device: str = "cpu",
                checkpoint: str | Path | None = None) -> list[dict[str, float]]:
    """Train cross-entropy classifier and retain the lowest-validation-loss checkpoint."""
    if epochs < 1 or lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    target = torch.device(device)
    model.to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        train_loss, train_n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(target), labels.to(target, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.numel()
            train_n += labels.numel()
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(target), labels.to(target, dtype=torch.long)
                val_loss += loss_fn(model(images), labels).item() * labels.numel()
                val_n += labels.numel()
        if train_n == 0 or val_n == 0:
            raise ValueError("Training and validation loaders must not be empty")
        row = {"epoch": float(epoch + 1), "train_loss": train_loss / train_n,
               "val_loss": val_loss / val_n, "epoch_seconds": time.perf_counter() - start}
        history.append(row)
        if row["val_loss"] < best:
            best = row["val_loss"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            if checkpoint is not None:
                save_checkpoint(model, checkpoint, metadata={"epoch": epoch + 1, "val_loss": best})
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_checkpoint(model: nn.Module, path: str | Path, *, metadata: dict | None = None) -> None:
    """Save weights and JSON-like metadata; never serializes arbitrary Python objects."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, destination)


def load_checkpoint(model: nn.Module, path: str | Path, *, map_location: str = "cpu") -> dict:
    """Load a state-dict checkpoint with safe deserialization where supported."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only support.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Invalid checkpoint: expected a state_dict payload")
    model.load_state_dict(payload["state_dict"])
    return dict(payload.get("metadata", {}))
