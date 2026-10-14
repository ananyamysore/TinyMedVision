"""Portable run artifacts: machine-readable metrics, manifest, and figures."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch

matplotlib.use("Agg")

def write_report(output_dir: str | Path, *, results: list[dict], config: dict,
                 notes: list[str] | None = None) -> None:
    """Write results CSV, config/manifest JSON, and a comparative metric figure."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "results.csv", index=False)
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    manifest = {"created_utc": datetime.now(UTC).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "device": config.get("device", "cpu"),
                "seed": config.get("seed"), "dataset": config.get("dataset"),
                "config_file": "config.json", "results_file": "results.csv",
                "notes": notes or [], "checkpoint_files": config.get("checkpoints", [])}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    metric = "roc_auc" if "roc_auc" in frame else "accuracy"
    if metric in frame and "model" in frame:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = pd.to_numeric(frame[metric], errors="coerce")
        ax.bar(frame["model"], values)
        ax.set(ylabel=metric.replace("_", " ").title(), ylim=(0, 1),
               title="Model comparison on held-out split")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        fig.savefig(output / "metrics.png", dpi=150)
        plt.close(fig)

