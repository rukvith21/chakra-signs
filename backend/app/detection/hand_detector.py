from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

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


class HandDetector:
    """Wraps MediaPipe Hands for real-time landmark detection."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
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
        rgb.flags.writeable = False
        self._raw_results = self._hands.process(rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        hands: list[HandData] = []
        raw = self._raw_results
        if raw.multi_hand_landmarks and raw.multi_handedness:
            for lm, hn in zip(raw.multi_hand_landmarks, raw.multi_handedness):
                cls = hn.classification[0]
                landmarks = [(p.x, p.y, p.z) for p in lm.landmark]
                hands.append(
                    HandData(
                        landmarks=landmarks,
                        handedness=cls.label,
                        confidence=cls.score,
                    )
                )

        return DetectionResult(
            hands=hands,
            num_hands=len(hands),
            processing_time_ms=elapsed_ms,
        )

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw MediaPipe landmarks and connections onto *frame* in-place."""
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

    def close(self) -> None:
        self._hands.close()
        logger.info("HandDetector closed")
