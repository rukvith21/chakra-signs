from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.core.config import load_app_config, load_config


class ConfigLoadingTests(unittest.TestCase):
    def test_load_app_config_uses_defaults_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing-config.json"
            cfg = load_app_config(missing_path)

        self.assertEqual(cfg.detection.max_num_hands, 2)
        self.assertAlmostEqual(cfg.detection.min_detection_confidence, 0.7)
        self.assertAlmostEqual(cfg.detection.min_tracking_confidence, 0.5)
        self.assertEqual(cfg.detection.model_complexity, 1)
        self.assertEqual(cfg.camera.index, 0)
        self.assertEqual(cfg.camera.width, 1280)
        self.assertEqual(cfg.camera.height, 720)
        self.assertTrue(cfg.camera.mirror)

    def test_load_config_returns_empty_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{bad json", encoding="utf-8")
            loaded = load_config(config_path)

        self.assertEqual(loaded, {})

    def test_load_app_config_sanitizes_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "detection": {
                            "max_num_hands": 42,
                            "min_detection_confidence": "oops",
                            "min_tracking_confidence": -3.0,
                            "model_complexity": 9,
                        },
                        "camera": {
                            "index": -2,
                            "width": 0,
                            "height": "bad",
                            "mirror": "no",
                        },
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_app_config(config_path)

        self.assertEqual(cfg.detection.max_num_hands, 2)
        self.assertAlmostEqual(cfg.detection.min_detection_confidence, 0.7)
        self.assertAlmostEqual(cfg.detection.min_tracking_confidence, 0.0)
        self.assertEqual(cfg.detection.model_complexity, 2)
        self.assertEqual(cfg.camera.index, 0)
        self.assertEqual(cfg.camera.width, 1)
        self.assertEqual(cfg.camera.height, 720)
        self.assertFalse(cfg.camera.mirror)


if __name__ == "__main__":
    unittest.main()
