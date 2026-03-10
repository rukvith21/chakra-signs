from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass
class DetectionConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    model_asset_path: str | None = None


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True


@dataclass(frozen=True)
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)


def _as_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_float(
    value: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load project configuration from a JSON file."""
    target_path = config_path or CONFIG_PATH
    if target_path.exists():
        try:
            with open(target_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in config file: %s. Using defaults.", target_path)
            return {}
        if not isinstance(cfg, dict):
            logger.warning("Config file %s must contain a JSON object. Using defaults.", target_path)
            return {}
        logger.info("Loaded config from %s", target_path)
        return cfg
    logger.info("Config not found at %s - using defaults", target_path)
    return {}


def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Return validated typed config with sane defaults."""
    raw = load_config(config_path)

    raw_detection = raw.get("detection", {})
    if not isinstance(raw_detection, dict):
        raw_detection = {}
    raw_camera = raw.get("camera", {})
    if not isinstance(raw_camera, dict):
        raw_camera = {}

    detection = DetectionConfig(
        max_num_hands=_as_int(raw_detection.get("max_num_hands"), 2, minimum=1, maximum=2),
        min_detection_confidence=_as_float(raw_detection.get("min_detection_confidence"), 0.7, minimum=0.0, maximum=1.0),
        min_hand_presence_confidence=_as_float(
            raw_detection.get("min_hand_presence_confidence"), 0.5, minimum=0.0, maximum=1.0
        ),
        min_tracking_confidence=_as_float(raw_detection.get("min_tracking_confidence"), 0.5, minimum=0.0, maximum=1.0),
        model_complexity=_as_int(raw_detection.get("model_complexity"), 1, minimum=0, maximum=2),
        model_asset_path=(
            str(raw_detection.get("model_asset_path"))
            if raw_detection.get("model_asset_path") not in (None, "")
            else None
        ),
    )
    camera = CameraConfig(
        index=_as_int(raw_camera.get("index"), 0, minimum=0),
        width=_as_int(raw_camera.get("width"), 1280, minimum=1),
        height=_as_int(raw_camera.get("height"), 720, minimum=1),
        mirror=_as_bool(raw_camera.get("mirror"), True),
    )
    return AppConfig(detection=detection, camera=camera)
