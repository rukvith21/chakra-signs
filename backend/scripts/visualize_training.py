#!/usr/bin/env python3
"""Visualize training data and metrics using Matplotlib.

Usage (from project root):
    python backend/scripts/visualize_training.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from backend.app.training.dataset_manager import DatasetManager


METRICS_DIR = _PROJECT_ROOT / "backend" / "data" / "metrics"
METRICS_FILE = METRICS_DIR / "train_metrics.json"


def _plot_class_distribution(counts: dict[str, int], out_path: Path) -> None:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, values, color="#3f7ae0")
    ax.set_title("Class Distribution (Recorded Samples)")
    ax.set_xlabel("Gesture Label")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", rotation=35)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(metrics: dict, out_path: Path) -> None:
    labels = metrics.get("labels", [])
    cm = np.array(metrics.get("confusion_matrix", []), dtype=np.int32)
    if cm.size == 0:
        raise ValueError("confusion_matrix missing or empty in train_metrics.json")

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix (Test Split)",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    manager = DatasetManager()
    counts = manager.label_counts()
    if not counts:
        print("No dataset samples found. Record data before visualization.")
        sys.exit(1)

    class_plot = METRICS_DIR / "class_distribution.png"
    _plot_class_distribution(counts, class_plot)
    print(f"Saved class distribution plot: {class_plot}")

    if not METRICS_FILE.exists():
        print("No training metrics file found. Run train_model.py first.")
        sys.exit(1)

    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    cm_plot = METRICS_DIR / "confusion_matrix.png"
    _plot_confusion_matrix(metrics, cm_plot)
    print(f"Saved confusion matrix plot: {cm_plot}")

    print("Visualization complete.")


if __name__ == "__main__":
    main()
