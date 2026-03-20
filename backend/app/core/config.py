from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "backend" / "data" / "models" / "hand_landmarker.task"


@dataclass
class DetectionConfig:
    model_path: str = str(DEFAULT_MODEL_PATH)
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def load_config() -> dict[str, Any]:
    """Load project configuration from config.json at the project root."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg: dict[str, Any] = json.load(f)
        logger.info("Loaded config from %s", CONFIG_PATH)
        return cfg
    logger.warning("Config not found at %s — using defaults", CONFIG_PATH)
    return {}
