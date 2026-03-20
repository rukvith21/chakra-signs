from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from backend.app.core.config import PROJECT_ROOT

_MODELS_DIR = PROJECT_ROOT / "backend" / "data" / "models"


class GestureClassifier:
    """Loads trained model artifacts and predicts gesture labels."""

    def __init__(
        self,
        model_path: Path | None = None,
        encoder_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or (_MODELS_DIR / "gesture_clf.joblib")
        self.encoder_path = encoder_path or (_MODELS_DIR / "label_encoder.joblib")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder file not found: {self.encoder_path}")

        self._clf = joblib.load(self.model_path)
        self._encoder = joblib.load(self.encoder_path)

    def predict(self, features: list[float]) -> tuple[str, float]:
        """Predict a gesture label from a single feature vector.

        Returns `(label, confidence)` where confidence is the top class score.
        """
        x = np.array([features], dtype=np.float32)
        pred_idx = int(self._clf.predict(x)[0])
        label = str(self._encoder.inverse_transform([pred_idx])[0])

        confidence = 1.0
        if hasattr(self._clf, "predict_proba"):
            probs = self._clf.predict_proba(x)[0]
            confidence = float(np.max(probs))

        return label, confidence
