from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.core.schema import GestureSample
from backend.app.training.gesture_trainer import GestureTrainer


def _make_landmarks(offset: float) -> list[tuple[float, float, float]]:
    return [(offset + i * 0.01, offset + i * 0.02, offset + i * 0.03) for i in range(21)]


def _make_sample(label: str, session_id: str, offset: float) -> GestureSample:
    landmarks = _make_landmarks(offset)
    return GestureSample(
        gesture_label=label,
        handedness="Right",
        normalized_landmarks=landmarks,
        raw_landmarks=landmarks,
        num_hands=1,
        session_id=session_id,
    )


class GestureTrainerTests(unittest.TestCase):
    def test_prepare_split_prefers_session_holdout_when_possible(self) -> None:
        samples: list[GestureSample] = []
        for session_id, base in [("s1", 0.0), ("s2", 1.0), ("s3", 2.0)]:
            for idx in range(4):
                samples.append(_make_sample("bird", session_id, base + idx * 0.01))
                samples.append(_make_sample("dog", session_id, base + 5.0 + idx * 0.01))

        trainer = GestureTrainer()
        prepared = trainer.prepare_split(samples, test_size=0.33, random_state=7, split_mode="session")

        self.assertEqual(prepared.split_strategy, "session_group_holdout")
        self.assertTrue(set(prepared.train_groups).isdisjoint(prepared.test_groups))
        self.assertEqual(set(prepared.tested_labels), {"bird", "dog"})
        self.assertEqual(prepared.missing_test_labels, [])

    def test_prepare_split_falls_back_when_group_holdout_is_impossible(self) -> None:
        samples = [
            _make_sample("bird", "s1", 0.0),
            _make_sample("bird", "s1", 0.1),
            _make_sample("bird", "s1", 0.2),
            _make_sample("dog", "s2", 5.0),
            _make_sample("dog", "s2", 5.1),
            _make_sample("dog", "s2", 5.2),
        ]

        trainer = GestureTrainer()
        prepared = trainer.prepare_split(samples, test_size=0.33, random_state=7, split_mode="session")

        self.assertEqual(prepared.split_strategy, "frame_stratified_fallback")

    def test_train_mlp_writes_model_artifacts(self) -> None:
        samples: list[GestureSample] = []
        for session_id, base in [("s1", 0.0), ("s2", 1.0), ("s3", 2.0)]:
            for idx in range(10):
                samples.append(_make_sample("bird", session_id, base + idx * 0.01))
                samples.append(_make_sample("dog", session_id, base + 5.0 + idx * 0.01))

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            metrics_dir = Path(tmp) / "metrics"
            trainer = GestureTrainer(models_dir=models_dir, metrics_dir=metrics_dir)

            result = trainer.train(
                samples=samples,
                test_size=0.2,
                random_state=7,
                model_type="mlp",
                split_mode="session",
                mlp_hidden_layers=(16,),
                mlp_max_iter=50,
            )

            self.assertEqual(result.model_type, "mlp")
            self.assertTrue(Path(result.model_path).exists())
            self.assertTrue(Path(result.encoder_path).exists())
            self.assertTrue(Path(result.metrics_path).exists())


if __name__ == "__main__":
    unittest.main()
