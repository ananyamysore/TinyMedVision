"""Compact grayscale image classifiers with a common logits interface."""

from __future__ import annotations

import torch
from torch import nn


class CompactCNN(nn.Module):
    """Small conventional CNN baseline."""

    def __init__(self, num_classes: int = 2, width: int = 16):
        super().__init__()
        if num_classes < 2 or width < 1:
            raise ValueError("num_classes must be >=2 and width positive")
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class DepthwiseCNN(nn.Module):
    """Depthwise-separable CNN with fewer multiply-adds at comparable widths."""

    def __init__(self, num_classes: int = 2, width: int = 16):
        super().__init__()
        if num_classes < 2 or width < 1:
            raise ValueError("num_classes must be >=2 and width positive")
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.Conv2d(width, width * 2, 1, bias=False), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class CompactResidualCNN(nn.Module):
    """Compact residual classifier."""

    def __init__(self, num_classes: int = 2, width: int = 16):
        super().__init__()
        if num_classes < 2 or width < 1:
            raise ValueError("num_classes must be >=2 and width positive")
        self.features = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(),
                                      ResidualBlock(width), nn.MaxPool2d(2),
                                      nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(),
                                      ResidualBlock(width * 2), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def build_model(name: str, *, num_classes: int = 2, width: int = 16) -> nn.Module:
    """Construct one supported architecture by stable name."""
    choices = {"compact_cnn": CompactCNN, "depthwise_cnn": DepthwiseCNN,
               "residual_cnn": CompactResidualCNN}
    try:
        return choices[name](num_classes=num_classes, width=width)
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(choices)}") from exc
