from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from backend.app.core.config import DetectionConfig

logger = logging.getLogger(__name__)

try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except Exception:  # pragma: no cover - import compatibility fallback
    mp_tasks = None
    mp_vision = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TASK_MODEL_PATH = PROJECT_ROOT / "backend" / "data" / "models" / "hand_landmarker.task"
HAND_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class HandData:
    """Processed data for a single detected hand."""

    landmarks: list[tuple[float, float, float]]
    handedness: str
    confidence: float

    def to_flat_array(self) -> np.ndarray:
        """Return landmarks as a 1-D array of length 63 (21 joints x 3)."""
        return np.array(self.landmarks, dtype=np.float32).flatten()


@dataclass
class DetectionResult:
    """Container for one frame's detection output."""

    hands: list[HandData]
    num_hands: int
    processing_time_ms: float


class HandDetector:
    """Wraps MediaPipe Hands for real-time landmark detection."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self._raw_results: Any = None
        self._connections: list[Any] = []
        self._last_timestamp_ms = 0
        self._backend = "legacy" if hasattr(mp, "solutions") else "tasks"

        if self._backend == "legacy":
            self._init_legacy_backend()
        else:
            self._init_tasks_backend()

    def _init_legacy_backend(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            model_complexity=self.config.model_complexity,
        )
        logger.info(
            "HandDetector ready (legacy) max_hands=%d det_conf=%.2f track_conf=%.2f",
            self.config.max_num_hands,
            self.config.min_detection_confidence,
            self.config.min_tracking_confidence,
        )

    def _init_tasks_backend(self) -> None:
        if mp_tasks is None or mp_vision is None:
            raise RuntimeError("MediaPipe Tasks backend is unavailable in this environment.")

        model_path = self._resolve_tasks_model_path()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.config.max_num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(options)
        self._connections = list(mp_vision.HandLandmarksConnections.HAND_CONNECTIONS)
        logger.info(
            "HandDetector ready (tasks) max_hands=%d det_conf=%.2f presence_conf=%.2f track_conf=%.2f model=%s",
            self.config.max_num_hands,
            self.config.min_detection_confidence,
            self.config.min_hand_presence_confidence,
            self.config.min_tracking_confidence,
            model_path,
        )

    def _resolve_tasks_model_path(self) -> Path:
        if self.config.model_asset_path:
            model_path = Path(self.config.model_asset_path)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
        else:
            model_path = DEFAULT_TASK_MODEL_PATH

        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Hand Landmarker model not found. Downloading to %s", model_path)
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_TASK_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download MediaPipe hand_landmarker.task model. "
                f"Please download it manually from {HAND_LANDMARKER_TASK_URL} "
                f"and place it at {model_path}."
            ) from exc
        return model_path

    @staticmethod
    def _extract_legacy_hands(raw: Any) -> list[HandData]:
        hands: list[HandData] = []
        if raw.multi_hand_landmarks and raw.multi_handedness:
            for lm, hn in zip(raw.multi_hand_landmarks, raw.multi_handedness):
                cls = hn.classification[0]
                landmarks = [(float(p.x), float(p.y), float(p.z)) for p in lm.landmark]
                hands.append(
                    HandData(
                        landmarks=landmarks,
                        handedness=cls.label,
                        confidence=float(cls.score),
                    )
                )
        return hands

    @staticmethod
    def _extract_tasks_hands(raw: Any) -> list[HandData]:
        hands: list[HandData] = []
        hand_landmarks = getattr(raw, "hand_landmarks", []) or []
        handedness_list = getattr(raw, "handedness", []) or []

        for idx, landmarks_per_hand in enumerate(hand_landmarks):
            handedness = "Unknown"
            confidence = 0.0
            if idx < len(handedness_list) and handedness_list[idx]:
                best = handedness_list[idx][0]
                handedness = best.category_name or best.display_name or "Unknown"
                confidence = float(best.score or 0.0)

            landmarks = [(float(p.x), float(p.y), float(p.z)) for p in landmarks_per_hand]
            hands.append(HandData(landmarks=landmarks, handedness=handedness, confidence=confidence))
        return hands
        self._raw_results = None
        logger.info(
            "HandDetector ready  max_hands=%d  det_conf=%.2f  track_conf=%.2f",
            self.config.max_num_hands,
            self.config.min_detection_confidence,
            self.config.min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run MediaPipe on a BGR frame, return structured results."""
        start = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._backend == "legacy":
            rgb.flags.writeable = False
            self._raw_results = self._hands.process(rgb)
            hands = self._extract_legacy_hands(self._raw_results)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = max(self._last_timestamp_ms + 1, int(time.perf_counter() * 1000))
            self._last_timestamp_ms = timestamp_ms
            self._raw_results = self._hands.detect_for_video(mp_image, timestamp_ms)
            hands = self._extract_tasks_hands(self._raw_results)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return DetectionResult(
            hands=hands,
            num_hands=len(hands),
            processing_time_ms=elapsed_ms,
        )

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw MediaPipe landmarks and connections onto *frame* in-place."""
        if self._backend == "legacy":
            if self._raw_results and self._raw_results.multi_hand_landmarks:
                for hand_lm in self._raw_results.multi_hand_landmarks:
                    self._mp_drawing.draw_landmarks(
                        frame,
                        hand_lm,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._mp_styles.get_default_hand_landmarks_style(),
                        self._mp_styles.get_default_hand_connections_style(),
                    )
            return frame

        hand_landmarks = getattr(self._raw_results, "hand_landmarks", []) if self._raw_results else []
        if not hand_landmarks:
            return frame

        h, w = frame.shape[:2]
        for landmarks_per_hand in hand_landmarks:
            points: list[tuple[int, int]] = []
            for lm in landmarks_per_hand:
                px = int(np.clip(lm.x, 0.0, 1.0) * (w - 1))
                py = int(np.clip(lm.y, 0.0, 1.0) * (h - 1))
                points.append((px, py))
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

            for connection in self._connections:
                start_idx = connection.start
                end_idx = connection.end
                if 0 <= start_idx < len(points) and 0 <= end_idx < len(points):
                    cv2.line(frame, points[start_idx], points[end_idx], (0, 180, 255), 2, lineType=cv2.LINE_AA)
        return frame

    def close(self) -> None:
        self._hands.close()
        logger.info("HandDetector closed")
