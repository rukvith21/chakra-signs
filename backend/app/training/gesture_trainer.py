from __future__ import annotations

import itertools
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.app.core.config import PROJECT_ROOT
from backend.app.core.schema import GestureSample
from backend.app.utils.gesture_features import sample_feature_vector

logger = logging.getLogger(__name__)

_MODELS_DIR = PROJECT_ROOT / "backend" / "data" / "models"
_METRICS_DIR = PROJECT_ROOT / "backend" / "data" / "metrics"

_MODEL_TYPES = {"random_forest", "mlp"}
_SESSION_SPLIT = "session_group_holdout"
_FRAME_SPLIT = "frame_stratified_fallback"


@dataclass
class PreparedSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    encoder: LabelEncoder
    split_strategy: str
    train_groups: list[str]
    test_groups: list[str]
    tested_labels: list[str]
    missing_test_labels: list[str]


@dataclass
class TrainResult:
    model_type: str
    split_strategy: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]
    labels: list[str]
    tested_labels: list[str]
    missing_test_labels: list[str]
    report: str
    model_path: str
    encoder_path: str
    metrics_path: str
    train_size: int
    test_size: int
    train_groups: list[str]
    test_groups: list[str]


def _samples_to_arrays(
    samples: list[GestureSample],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert samples to feature matrix X and string label array y."""
    feature_rows = [sample_feature_vector(sample) for sample in samples]
    sizes = {len(row) for row in feature_rows}
    if len(sizes) > 1:
        raise ValueError(
            f"Mixed feature sizes in dataset: {sizes}. "
            "Filter samples so all have the same landmark mode (e.g. all single-hand "
            "or all two-hand). Use --labels to train on specific gesture sets."
        )

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array([s.gesture_label for s in samples])
    return X, y


def _sample_groups(samples: list[GestureSample]) -> np.ndarray:
    """Use session IDs for grouped evaluation, falling back to unique sample IDs."""
    groups: list[str] = []
    for sample in samples:
        if sample.session_id:
            groups.append(sample.session_id)
        else:
            groups.append(f"sample:{sample.sample_id}")
    return np.array(groups, dtype=object)


def _artifact_name(model_type: str) -> str:
    if model_type == "random_forest":
        return "gesture_random_forest"
    if model_type == "mlp":
        return "gesture_mlp"
    raise ValueError(f"Unsupported model_type: {model_type}")


def _evaluate_group_candidate(
    test_groups: tuple[str, ...],
    group_indices: dict[str, np.ndarray],
    y_encoded: np.ndarray,
    all_label_ids: set[int],
    target_test_size: int,
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray] | None:
    test_idx = np.concatenate([group_indices[group] for group in test_groups])
    train_parts = [idx for group, idx in group_indices.items() if group not in test_groups]
    if not train_parts:
        return None

    train_idx = np.concatenate(train_parts)
    train_label_ids = {int(v) for v in np.unique(y_encoded[train_idx])}
    if train_label_ids != all_label_ids:
        return None

    test_label_ids = {int(v) for v in np.unique(y_encoded[test_idx])}
    if len(test_label_ids) < 2:
        return None

    score = (
        len(test_label_ids),
        -abs(len(test_idx) - target_test_size),
        -len(test_groups),
    )
    return score, train_idx, test_idx


def _group_holdout_indices(
    y_encoded: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a session-disjoint holdout that keeps every class in training."""
    unique_groups = sorted({str(group) for group in groups})
    if len(unique_groups) < 2:
        return None

    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }
    all_label_ids = {int(v) for v in np.unique(y_encoded)}
    target_test_size = max(1, int(round(len(y_encoded) * test_size)))

    best: tuple[tuple[int, int, int], np.ndarray, np.ndarray] | None = None

    if len(unique_groups) <= 16:
        candidates = itertools.chain.from_iterable(
            itertools.combinations(unique_groups, size)
            for size in range(1, len(unique_groups))
        )
    else:
        rng = np.random.default_rng(random_state)
        n_groups = len(unique_groups)
        target_groups = max(1, min(n_groups - 1, int(round(n_groups * test_size))))
        sampled: list[tuple[str, ...]] = []
        for _ in range(1024):
            jitter = int(rng.integers(-1, 2))
            sample_size = max(1, min(n_groups - 1, target_groups + jitter))
            choice = tuple(sorted(rng.choice(unique_groups, size=sample_size, replace=False).tolist()))
            sampled.append(choice)
        candidates = sampled

    for test_groups in candidates:
        candidate = _evaluate_group_candidate(
            test_groups=test_groups,
            group_indices=group_indices,
            y_encoded=y_encoded,
            all_label_ids=all_label_ids,
            target_test_size=target_test_size,
        )
        if candidate is None:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None

    _, train_idx, test_idx = best
    return train_idx, test_idx


def _labels_from_indices(encoder: LabelEncoder, encoded_labels: np.ndarray) -> list[str]:
    if len(encoded_labels) == 0:
        return []
    unique = np.unique(encoded_labels)
    return [str(label) for label in encoder.inverse_transform(unique)]


class GestureTrainer:
    """Train, evaluate, and save static gesture classifiers."""

    def __init__(
        self,
        models_dir: Path | None = None,
        metrics_dir: Path | None = None,
    ) -> None:
        self.models_dir = models_dir or _MODELS_DIR
        self.metrics_dir = metrics_dir or _METRICS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def prepare_split(
        self,
        samples: list[GestureSample],
        test_size: float = 0.2,
        random_state: int = 42,
        split_mode: str = "session",
    ) -> PreparedSplit:
        labels = sorted(set(s.gesture_label for s in samples))
        if len(labels) < 2:
            raise ValueError(
                f"Need at least 2 gesture classes to train, found {len(labels)}: {labels}. "
                "Record more gestures with different labels using record_gestures.py."
            )

        X, y = _samples_to_arrays(samples)
        groups = _sample_groups(samples)
        logger.info("Dataset: %d samples, %d features, %d classes", *X.shape, len(labels))

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        split_strategy = _FRAME_SPLIT
        if split_mode not in {"session", "frame"}:
            raise ValueError(f"Unsupported split_mode: {split_mode}")

        grouped_split = None
        if split_mode == "session":
            grouped_split = _group_holdout_indices(
                y_encoded=y_encoded,
                groups=groups,
                test_size=test_size,
                random_state=random_state,
            )

        if grouped_split is not None:
            train_idx, test_idx = grouped_split
            split_strategy = _SESSION_SPLIT
            logger.info(
                "Using session-aware split with %d train sessions and %d test sessions",
                len({groups[i] for i in train_idx}),
                len({groups[i] for i in test_idx}),
            )
        else:
            if split_mode == "session":
                logger.warning(
                    "No valid session-aware holdout preserved all training labels; "
                    "falling back to frame-level stratified split. Record more independent "
                    "sessions per label for a cleaner evaluation."
                )
            train_idx, test_idx = train_test_split(
                np.arange(len(samples)),
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded,
            )

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        tested_labels = _labels_from_indices(encoder, y_test)
        missing_test_labels = sorted(str(label) for label in set(encoder.classes_) - set(tested_labels))

        return PreparedSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            encoder=encoder,
            split_strategy=split_strategy,
            train_groups=sorted({str(groups[i]) for i in train_idx}),
            test_groups=sorted({str(groups[i]) for i in test_idx}),
            tested_labels=tested_labels,
            missing_test_labels=missing_test_labels,
        )

    def _build_model(
        self,
        model_type: str,
        random_state: int,
        n_estimators: int,
        mlp_hidden_layers: tuple[int, ...],
        mlp_max_iter: int,
    ) -> Any:
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )

        if model_type == "mlp":
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=mlp_hidden_layers,
                            max_iter=mlp_max_iter,
                            early_stopping=True,
                            n_iter_no_change=20,
                            random_state=random_state,
                        ),
                    ),
                ]
            )

        raise ValueError(f"Unsupported model_type: {model_type}")

    def train_from_split(
        self,
        prepared: PreparedSplit,
        model_type: str = "random_forest",
        random_state: int = 42,
        n_estimators: int = 100,
        mlp_hidden_layers: tuple[int, ...] = (128, 64),
        mlp_max_iter: int = 400,
    ) -> TrainResult:
        if model_type not in _MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {model_type}")

        clf = self._build_model(
            model_type=model_type,
            random_state=random_state,
            n_estimators=n_estimators,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_max_iter=mlp_max_iter,
        )
        clf.fit(prepared.X_train, prepared.y_train)

        y_pred = clf.predict(prepared.X_test)
        label_ids = np.arange(len(prepared.encoder.classes_))
        target_names = [str(label) for label in prepared.encoder.classes_]

        acc = accuracy_score(prepared.y_test, y_pred)
        prec = precision_score(prepared.y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(prepared.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(prepared.y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(prepared.y_test, y_pred, labels=label_ids).tolist()
        report = classification_report(
            prepared.y_test,
            y_pred,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
        )

        stem = _artifact_name(model_type)
        model_path = self.models_dir / f"{stem}.joblib"
        encoder_path = self.models_dir / f"label_encoder_{model_type}.joblib"
        metrics_path = self.metrics_dir / f"train_metrics_{model_type}.json"

        joblib.dump(clf, model_path)
        joblib.dump(prepared.encoder, encoder_path)

        # Keep the original file names as aliases for the default random forest path.
        if model_type == "random_forest":
            joblib.dump(clf, self.models_dir / "gesture_clf.joblib")
            joblib.dump(prepared.encoder, self.models_dir / "label_encoder.joblib")

        metrics: dict[str, Any] = {
            "model_type": model_type,
            "split_strategy": prepared.split_strategy,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "labels": target_names,
            "tested_labels": prepared.tested_labels,
            "missing_test_labels": prepared.missing_test_labels,
            "train_groups": prepared.train_groups,
            "test_groups": prepared.test_groups,
            "train_size": len(prepared.X_train),
            "test_size": len(prepared.X_test),
            "n_features": int(prepared.X_train.shape[1]),
            "n_estimators": n_estimators if model_type == "random_forest" else None,
            "mlp_hidden_layers": list(mlp_hidden_layers) if model_type == "mlp" else None,
            "mlp_max_iter": mlp_max_iter if model_type == "mlp" else None,
            "random_state": random_state,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Saved %s model to %s", model_type, model_path)

        return TrainResult(
            model_type=model_type,
            split_strategy=prepared.split_strategy,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            confusion=cm,
            labels=target_names,
            tested_labels=prepared.tested_labels,
            missing_test_labels=prepared.missing_test_labels,
            report=report,
            model_path=str(model_path),
            encoder_path=str(encoder_path),
            metrics_path=str(metrics_path),
            train_size=len(prepared.X_train),
            test_size=len(prepared.X_test),
            train_groups=prepared.train_groups,
            test_groups=prepared.test_groups,
        )

    def train(
        self,
        samples: list[GestureSample],
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        model_type: str = "random_forest",
        split_mode: str = "session",
        mlp_hidden_layers: tuple[int, ...] = (128, 64),
        mlp_max_iter: int = 400,
    ) -> TrainResult:
        prepared = self.prepare_split(
            samples=samples,
            test_size=test_size,
            random_state=random_state,
            split_mode=split_mode,
        )
        return self.train_from_split(
            prepared=prepared,
            model_type=model_type,
            random_state=random_state,
            n_estimators=n_estimators,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_max_iter=mlp_max_iter,
        )
