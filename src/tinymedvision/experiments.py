"""Controlled experiments for synthetic smoke evaluation and manifest datasets."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .calibration import calibration_metrics
from .data import ManifestDataset, read_manifest, validate_samples
from .efficiency import benchmark_model
from .evaluation import bootstrap_auc_ci, evaluate_predictions
from .models import build_model
from .reporting import write_report
from .robustness import evaluate_corruptions
from .splitting import split_indices, validate_split_groups
from .training import seed_everything, train_model


def run_experiment(manifest_path: str | Path, output_dir: str | Path, *, seed: int = 17,
                   epochs: int = 2, batch_size: int = 16, image_size: int = 28,
                   models: tuple[str, ...] = ("compact_cnn", "depthwise_cnn", "residual_cnn"),
                   device: str = "cpu", robustness: bool = False) -> list[dict]:
    """Train/evaluate same split and budget for each named model; test is touched once."""
    seed_everything(seed)
    samples = read_manifest(manifest_path)
    validate_samples(samples, expected_classes={0, 1})
    labels = np.asarray([sample.label for sample in samples])
    groups = [sample.group_id for sample in samples]
    group_values = groups if all(groups) else None
    declared = [sample.split for sample in samples]
    if any(declared):
        if not all(declared):
            raise ValueError("If split is provided, every manifest row must declare a split")
        validate_split_groups(samples)
        normalized = [str(value).lower() for value in declared]
        train_idx = np.asarray([i for i, value in enumerate(normalized)
                                if value in {"train", "training"}], dtype=int)
        val_idx = np.asarray([i for i, value in enumerate(normalized)
                              if value in {"val", "valid", "validation"}], dtype=int)
        test_idx = np.asarray([i for i, value in enumerate(normalized)
                               if value in {"test", "testing"}], dtype=int)
        if not train_idx.size or not test_idx.size:
            raise ValueError("Declared splits require at least one train and one test sample")
    else:
        train_idx, test_idx = split_indices(labels, groups=group_values, seed=seed)
        val_idx = np.asarray([], dtype=int)
    dataset = ManifestDataset(samples, size=(image_size, image_size))
    # If no validation partition was declared, reserve grouped records from training.
    if val_idx.size == 0:
        train_labels = labels[train_idx]
        train_groups = [groups[i] for i in train_idx] if group_values is not None else None
        core_local, val_local = split_indices(train_labels, test_size=0.2,
                                              groups=train_groups, seed=seed + 1)
        core_global = train_idx[core_local].tolist()
        val_global = train_idx[val_local].tolist()
    else:
        core_global = train_idx.tolist()
        val_global = val_idx.tolist()
    train_loader = DataLoader(Subset(dataset, core_global), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(Subset(dataset, val_global), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=batch_size, shuffle=False)
    out = Path(output_dir)
    results: list[dict] = []
    checkpoint_files = []
    for name in models:
        seed_everything(seed)
        model = build_model(name)
        checkpoint = out / "checkpoints" / f"{name}.pt"
        start = time.perf_counter()
        history = train_model(model, train_loader, val_loader, epochs=epochs, device=device,
                              checkpoint=checkpoint)
        train_seconds = time.perf_counter() - start
        y, probabilities = [], []
        model.eval()
        with torch.inference_mode():
            for images, target in test_loader:
                logits = model(images.to(device))
                probabilities.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
                y.extend(target.tolist())
        # Prespecified 0.5 threshold; do not select thresholds on held-out test labels.
        metrics = evaluate_predictions(y, probabilities, threshold=0.5)
        groups_test = [groups[i] for i in test_idx.tolist()] if group_values else None
        auc_ci = bootstrap_auc_ci(y, probabilities, groups=groups_test, seed=seed, n_bootstrap=200)
        calibration = calibration_metrics(y, probabilities)
        efficiency = benchmark_model(model, input_shape=(1, image_size, image_size),
                                     iterations=20, warmup=3, device=device, checkpoint=checkpoint)
        row = {"model": name, "condition": "clean", **metrics,
               "auc_ci_low": auc_ci[0] if auc_ci else None,
               "auc_ci_high": auc_ci[1] if auc_ci else None,
               "brier": calibration["brier"], "ece": calibration["ece"],
               "train_seconds": train_seconds, "best_epoch_val_loss": min(x["val_loss"] for x in history),
               **efficiency}
        results.append(row)
        if robustness:
            test_images = torch.cat([images for images, _ in test_loader])
            test_labels = torch.cat([target for _, target in test_loader])
            for stress in evaluate_corruptions(model, test_images, test_labels,
                                               seed=seed, device=device):
                results.append({"model": name, "condition": stress["corruption"], **stress})
        checkpoint_files.append(str(checkpoint.relative_to(out)))
    config = {"seed": seed, "dataset": str(Path(manifest_path).resolve()), "models": list(models),
              "epochs": epochs, "batch_size": batch_size, "image_size": image_size,
              "threshold": 0.5, "split_seed": seed, "device": device,
              "robustness": robustness,
              "python": platform.python_version(), "torch": torch.__version__,
              "split": {"train_n": len(core_global), "validation_n": len(val_global),
                        "test_n": len(test_idx), "grouped": group_values is not None,
                        "declared_splits_used": any(declared)},
              "checkpoints": checkpoint_files}
    write_report(out, results=results, config=config,
                 notes=["Synthetic data demonstrates software behavior only; not clinical evidence.",
                        "The default threshold is prespecified at 0.5."])
    return results
