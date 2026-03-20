from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from backend.app.core.config import PROJECT_ROOT
from backend.app.core.schema import GestureSample
from backend.app.training.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

_MODELS_DIR = PROJECT_ROOT / "backend" / "data" / "models"
_METRICS_DIR = PROJECT_ROOT / "backend" / "data" / "metrics"
_PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"


@dataclass
class TrainResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]
    labels: list[str]
    report: str
    model_path: str
    encoder_path: str
    metrics_path: str
    train_size: int
    test_size: int


def _flatten_landmarks(landmarks: list[tuple[float, float, float]]) -> list[float]:
    """Flatten 21 (x,y,z) landmarks into a 63-element feature vector."""
    features: list[float] = []
    for x, y, z in landmarks:
        features.extend([x, y, z])
    return features


def _samples_to_arrays(
    samples: list[GestureSample],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert samples to feature matrix X and label array y.

    All samples must have the same landmark count (21 for single-hand,
    42 for two-hand Naruto signs, etc.) so the feature matrix is rectangular.
    """
    sizes = {len(s.normalized_landmarks) for s in samples}
    if len(sizes) > 1:
        raise ValueError(
            f"Mixed landmark counts in dataset: {sizes}. "
            "Filter samples so all have the same count (e.g. all single-hand "
            "or all two-hand). Use --labels to train on specific gesture sets."
        )
    X = np.array([_flatten_landmarks(s.normalized_landmarks) for s in samples])
    y = np.array([s.gesture_label for s in samples])
    return X, y


class GestureTrainer:
    """Train, evaluate, and save a static gesture classifier."""

    def __init__(
        self,
        models_dir: Path | None = None,
        metrics_dir: Path | None = None,
    ) -> None:
        self.models_dir = models_dir or _MODELS_DIR
        self.metrics_dir = metrics_dir or _METRICS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        samples: list[GestureSample],
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
    ) -> TrainResult:
        labels = sorted(set(s.gesture_label for s in samples))
        if len(labels) < 2:
            raise ValueError(
                f"Need at least 2 gesture classes to train, found {len(labels)}: {labels}. "
                "Record more gestures with different labels using record_gestures.py."
            )

        X, y = _samples_to_arrays(samples)
        logger.info("Dataset: %d samples, %d features, %d classes", *X.shape, len(labels))

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        target_names = list(encoder.classes_)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

        model_path = self.models_dir / "gesture_clf.joblib"
        encoder_path = self.models_dir / "label_encoder.joblib"
        joblib.dump(clf, model_path)
        joblib.dump(encoder, encoder_path)
        logger.info("Model saved to %s", model_path)

        metrics: dict[str, Any] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "labels": target_names,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_estimators": n_estimators,
            "random_state": random_state,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        metrics_path = self.metrics_dir / "train_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return TrainResult(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            confusion=cm,
            labels=target_names,
            report=report,
            model_path=str(model_path),
            encoder_path=str(encoder_path),
            metrics_path=str(metrics_path),
            train_size=len(X_train),
            test_size=len(X_test),
        )
