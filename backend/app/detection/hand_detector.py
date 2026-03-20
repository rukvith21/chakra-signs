from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from backend.app.core.config import DetectionConfig

logger = logging.getLogger(__name__)


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


_HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


class HandDetector:
    """Wraps MediaPipe HandLandmarker (tasks API) for real-time detection."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

        base_options = mp_python.BaseOptions(
            model_asset_path=self.config.model_path,
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_result: vision.HandLandmarkerResult | None = None
        self._frame_ts_ms: int = 0

        logger.info(
            "HandDetector ready  num_hands=%d  det_conf=%.2f  presence_conf=%.2f  track_conf=%.2f",
            self.config.num_hands,
            self.config.min_hand_detection_confidence,
            self.config.min_hand_presence_confidence,
            self.config.min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run HandLandmarker on a BGR frame, return structured results."""
        start = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 33  # ~30 fps monotonic timestamp
        raw = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        self._last_result = raw

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        hands: list[HandData] = []
        if raw.hand_landmarks and raw.handedness:
            for lm_list, hn_list in zip(raw.hand_landmarks, raw.handedness):
                top = hn_list[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in lm_list]
                hands.append(
                    HandData(
                        landmarks=landmarks,
                        handedness=top.category_name,
                        confidence=top.score,
                    )
                )

        return DetectionResult(
            hands=hands,
            num_hands=len(hands),
            processing_time_ms=elapsed_ms,
        )

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw hand landmarks and connections onto *frame* in-place."""
        if self._last_result and self._last_result.hand_landmarks:
            for lm_list in self._last_result.hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame,
                    lm_list,
                    _HAND_CONNECTIONS,
                )
        return frame

    def close(self) -> None:
        self._landmarker.close()
        logger.info("HandDetector closed")
