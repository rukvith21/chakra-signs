#!/usr/bin/env python3
"""Train gesture classifiers on recorded landmark data.

Usage (from project root):
    python backend/scripts/train_model.py
    python backend/scripts/train_model.py --labels bird boar dog tiger snake rat
    python backend/scripts/train_model.py --model all --split-mode session
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.training.dataset_manager import DatasetManager
from backend.app.training.gesture_trainer import GestureTrainer, TrainResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger("train_model")


def _parse_hidden_layers(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Hidden layers cannot be empty.")

    try:
        layers = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Hidden layers must be comma-separated integers.") from exc

    if any(size <= 0 for size in layers):
        raise argparse.ArgumentTypeError("Hidden layer sizes must be positive integers.")
    return layers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gesture classifiers")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Only train on these gesture labels (for example: --labels bird tiger snake)",
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "mlp", "all"],
        default="random_forest",
        help="Which model backend to train.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["session", "frame"],
        default="session",
        help="Evaluation split. 'session' is stricter; 'frame' matches the old behavior.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples used for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits and model training.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees for the random forest backend.",
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        type=_parse_hidden_layers,
        default=(128, 64),
        help="Comma-separated hidden layer sizes for the neural net backend.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=400,
        help="Maximum MLP training iterations.",
    )
    return parser.parse_args()


def _load_samples(manager: DatasetManager, args: argparse.Namespace) -> tuple[list, dict[str, int]]:
    all_counts = manager.label_counts()
    if not all_counts:
        print("No gesture samples found.")
        print("Record gestures first:")
        print("  python backend/scripts/record_naruto.py")
        print("  python backend/scripts/record_gestures.py")
        sys.exit(1)

    if args.labels:
        missing = [label for label in args.labels if label not in all_counts]
        if missing:
            print(f"WARNING: Labels not found in dataset: {missing}")
        counts = {label: n for label, n in all_counts.items() if label in args.labels}
    else:
        counts = all_counts

    if len(counts) < 2:
        print(
            "\nYou need at least 2 different gesture classes to train.\n"
            "Record more gestures with different labels:\n"
            "  python backend/scripts/record_naruto.py\n"
            "  python backend/scripts/record_gestures.py\n"
        )
        sys.exit(1)

    if args.labels:
        samples = []
        for label in args.labels:
            samples.extend(manager.load_label(label))
    else:
        samples = manager.load_all()

    return samples, counts


def _print_dataset_summary(counts: dict[str, int], label_filter: list[str] | None) -> None:
    print("\n" + "=" * 64)
    print("  DATASET SUMMARY")
    if label_filter:
        print(f"  Filtered labels: {', '.join(label_filter)}")
    print("=" * 64)
    total = 0
    for label, n in sorted(counts.items()):
        print(f"  {label:20s}  {n:>5d} samples")
        total += n
    print(f"  {'TOTAL':20s}  {total:>5d} samples")
    print(f"  Classes: {len(counts)}")
    print("=" * 64)


def _print_result(result: TrainResult) -> None:
    print("\n" + "=" * 64)
    print(f"  RESULTS - {result.model_type}")
    print("=" * 64)
    print(f"  Split strategy: {result.split_strategy}")
    print(f"  Train set:      {result.train_size} samples")
    print(f"  Test set:       {result.test_size} samples")
    print(f"  Accuracy:       {result.accuracy:.4f}")
    print(f"  Precision:      {result.precision:.4f}")
    print(f"  Recall:         {result.recall:.4f}")
    print(f"  F1 Score:       {result.f1:.4f}")
    print(f"  Tested labels:  {', '.join(result.tested_labels) if result.tested_labels else '(none)'}")
    if result.missing_test_labels:
        print(f"  Not in test:    {', '.join(result.missing_test_labels)}")
    print(f"  Train groups:   {', '.join(result.train_groups)}")
    print(f"  Test groups:    {', '.join(result.test_groups)}")
    print(f"  Model saved:    {result.model_path}")
    print(f"  Encoder saved:  {result.encoder_path}")
    print(f"  Metrics saved:  {result.metrics_path}")
    print("=" * 64)
    print("\nClassification Report:")
    print(result.report)


def _print_comparison(results: list[TrainResult]) -> None:
    if len(results) < 2:
        return

    print("\n" + "=" * 64)
    print("  MODEL COMPARISON")
    print("=" * 64)
    print(f"  {'Model':16s} {'Accuracy':>10s} {'F1':>10s} {'Split':>24s}")
    for result in results:
        print(
            f"  {result.model_type:16s} "
            f"{result.accuracy:10.4f} "
            f"{result.f1:10.4f} "
            f"{result.split_strategy:>24s}"
        )
    best = max(results, key=lambda item: item.f1)
    print("=" * 64)
    print(f"  Best by F1: {best.model_type}")


def main() -> None:
    args = _parse_args()
    manager = DatasetManager()
    samples, counts = _load_samples(manager, args)

    _print_dataset_summary(counts, args.labels)
    print(f"\nLoaded {len(samples)} samples across {len(counts)} classes.")
    print(f"Split mode: {args.split_mode}")

    trainer = GestureTrainer()
    try:
        prepared = trainer.prepare_split(
            samples=samples,
            test_size=args.test_size,
            random_state=args.random_state,
            split_mode=args.split_mode,
        )
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    model_types = ["random_forest", "mlp"] if args.model == "all" else [args.model]
    results: list[TrainResult] = []
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        result = trainer.train_from_split(
            prepared=prepared,
            model_type=model_type,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            mlp_hidden_layers=args.mlp_hidden_layers,
            mlp_max_iter=args.mlp_max_iter,
        )
        results.append(result)
        _print_result(result)

    _print_comparison(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
