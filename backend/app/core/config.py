from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass
class DetectionConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1


def load_config() -> dict[str, Any]:
    """Load project configuration from config.json at the project root."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg: dict[str, Any] = json.load(f)
        logger.info("Loaded config from %s", CONFIG_PATH)
        return cfg
    logger.warning("Config not found at %s — using defaults", CONFIG_PATH)
    return {}
