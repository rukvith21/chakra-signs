#!/usr/bin/env python3
"""Phase 4 — Train a static gesture classifier on recorded landmark data.

Usage (from project root):
    python backend/scripts/train_model.py
    python backend/scripts/train_model.py --labels bird boar dog tiger snake rat

The --labels flag lets you train on a subset of gesture classes (useful when
mixing single-hand and two-hand data that have different landmark counts).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.training.dataset_manager import DatasetManager
from backend.app.training.gesture_trainer import GestureTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("train_model")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a gesture classifier")
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Only train on these gesture labels (e.g. --labels bird tiger snake)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manager = DatasetManager()
    all_counts = manager.label_counts()

    if not all_counts:
        print("No gesture samples found.")
        print("Record gestures first:")
        print("  python backend/scripts/record_naruto.py    (Naruto hand signs)")
        print("  python backend/scripts/record_gestures.py  (custom gestures)")
        sys.exit(1)

    if args.labels:
        missing = [l for l in args.labels if l not in all_counts]
        if missing:
            print(f"WARNING: Labels not found in dataset: {missing}")
        counts = {l: n for l, n in all_counts.items() if l in args.labels}
    else:
        counts = all_counts

    print("\n" + "=" * 56)
    print("  DATASET SUMMARY")
    if args.labels:
        print(f"  (filtered to: {', '.join(args.labels)})")
    print("=" * 56)
    total = 0
    for label, n in sorted(counts.items()):
        print(f"  {label:20s}  {n:>5d} samples")
        total += n
    print(f"  {'TOTAL':20s}  {total:>5d} samples")
    print(f"  Classes: {len(counts)}")
    print("=" * 56)

    if len(counts) < 2:
        print(
            "\nYou need at least 2 different gesture classes to train.\n"
            "Record more gestures with different labels:\n"
            "  python backend/scripts/record_naruto.py    (Naruto hand signs)\n"
            "  python backend/scripts/record_gestures.py  (custom gestures)\n"
            "\n"
            "Aim for at least 30+ samples per class for good results."
        )
        sys.exit(1)

    if args.labels:
        samples = []
        for label in args.labels:
            samples.extend(manager.load_label(label))
    else:
        samples = manager.load_all()
    print(f"\nLoaded {len(samples)} samples across {len(counts)} classes.")

    trainer = GestureTrainer()
    print("Training RandomForestClassifier...")
    result = trainer.train(samples)

    print("\n" + "=" * 56)
    print("  TRAINING RESULTS")
    print("=" * 56)
    print(f"  Train set:  {result.train_size} samples")
    print(f"  Test set:   {result.test_size} samples")
    print(f"  Accuracy:   {result.accuracy:.4f}")
    print(f"  Precision:  {result.precision:.4f}")
    print(f"  Recall:     {result.recall:.4f}")
    print(f"  F1 Score:   {result.f1:.4f}")
    print("=" * 56)

    print("\nClassification Report:")
    print(result.report)

    print("Confusion Matrix:")
    header = "  " + "  ".join(f"{l:>10s}" for l in result.labels)
    print(f"  {'Predicted →':>10s}{header}")
    for label, row in zip(result.labels, result.confusion):
        vals = "  ".join(f"{v:>10d}" for v in row)
        print(f"  {label:>10s}  {vals}")

    print(f"\nModel saved:    {result.model_path}")
    print(f"Encoder saved:  {result.encoder_path}")
    print(f"Metrics saved:  {result.metrics_path}")
    print("\nDone. You can now use this model for real-time inference (Phase 5).")


if __name__ == "__main__":
    main()
