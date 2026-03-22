#!/usr/bin/env python3
"""Phase 5 — Real-time Naruto sign inference with webcam.

Usage (from project root):
    python backend/scripts/run_inference.py

Controls:
    q  — quit
"""
from __future__ import annotations

import sys
import time
from collections import Counter, deque
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from backend.app.core.config import DetectionConfig, load_config
from backend.app.detection.hand_detector import DetectionResult, HandDetector
from backend.app.inference.gesture_classifier import GestureClassifier
from backend.app.utils.gesture_features import build_feature_vector
from backend.app.utils.hud import text_with_bg


class _TemporalSmoother:
    """Smooths frame-wise predictions into a stable label."""

    def __init__(self, window_size: int = 9, min_votes: int = 4) -> None:
        self.window_size = window_size
        self.min_votes = min_votes
        self._history: deque[str] = deque(maxlen=window_size)
        self._conf_history: deque[float] = deque(maxlen=window_size)

    def update(self, label: str, conf: float) -> tuple[str, float]:
        self._history.append(label)
        self._conf_history.append(conf)

        counts = Counter(self._history)
        top_label, top_votes = counts.most_common(1)[0]

        if top_votes < self.min_votes:
            # Not enough consensus yet; keep latest frame prediction.
            return label, conf

        confs = [c for l, c in zip(self._history, self._conf_history) if l == top_label]
        mean_conf = float(sum(confs) / max(len(confs), 1))
        return top_label, mean_conf


def _ordered_two_hands(
    result: DetectionResult,
    min_confidence: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]] | None:
    """Return (left, right) landmarks for two confident hands, else None."""
    good_hands = [h for h in result.hands if h.confidence >= min_confidence]
    if len(good_hands) < 2:
        return None

    left_hand = None
    right_hand = None

    for hand in good_hands:
        if hand.handedness == "Left" and left_hand is None:
            left_hand = hand
        elif hand.handedness == "Right" and right_hand is None:
            right_hand = hand

    if left_hand is None or right_hand is None:
        h0, h1 = good_hands[0], good_hands[1]
        cx0 = float(np.mean([lm[0] for lm in h0.landmarks]))
        cx1 = float(np.mean([lm[0] for lm in h1.landmarks]))
        if cx0 < cx1:
            left_hand, right_hand = h0, h1
        else:
            left_hand, right_hand = h1, h0

    return left_hand.landmarks, right_hand.landmarks


def _features_from_two_hands(
    left_raw: list[tuple[float, float, float]],
    right_raw: list[tuple[float, float, float]],
) -> list[float]:
    return build_feature_vector(list(left_raw) + list(right_raw))


def _draw_hud(
    frame: np.ndarray,
    result: DetectionResult,
    fps: float,
    pred_label: str,
    pred_conf: float,
    status_text: str,
    status_color: tuple[int, int, int],
) -> np.ndarray:
    h, w = frame.shape[:2]

    text_with_bg(frame, f"FPS: {fps:.0f}", (12, 30), 0.70, (0, 255, 100))
    text_with_bg(frame, f"Hands: {result.num_hands}", (12, 64), 0.70, (0, 255, 100))

    hand_colors = {"Left": (255, 180, 0), "Right": (0, 220, 0)}
    for i, hand in enumerate(result.hands):
        color = hand_colors.get(hand.handedness, (255, 255, 255))
        y = 104 + i * 36
        text_with_bg(frame, f"{hand.handedness} | conf {hand.confidence:.0%}", (12, y), 0.62, color)

    text_with_bg(frame, f"Pred: {pred_label}", (w - 360, 30), 0.95, (0, 255, 255))
    text_with_bg(frame, f"Conf: {pred_conf:.0%}", (w - 360, 70), 0.70, (220, 220, 220))
    text_with_bg(frame, status_text, (w - 360, 104), 0.58, status_color)

    text_with_bg(
        frame,
        f"Detection: {result.processing_time_ms:.1f} ms",
        (12, h - 44),
        0.52,
        (180, 180, 180),
    )
    text_with_bg(frame, "q: quit", (12, h - 14), 0.50, (160, 160, 160))
    return frame


def main() -> None:
    cfg = load_config()
    det_cfg = cfg.get("detection", {})
    cam_cfg = cfg.get("camera", {})
    rec_cfg = cfg.get("recording", {})
    inf_cfg = cfg.get("inference", {})

    detection_config = DetectionConfig(
        num_hands=det_cfg.get("num_hands", 2),
        min_hand_detection_confidence=det_cfg.get("min_hand_detection_confidence", 0.7),
        min_hand_presence_confidence=det_cfg.get("min_hand_presence_confidence", 0.5),
        min_tracking_confidence=det_cfg.get("min_tracking_confidence", 0.5),
    )

    model_type = str(inf_cfg.get("model_type", "random_forest"))

    detector = HandDetector(detection_config)
    classifier = GestureClassifier(model_type=model_type)
    min_conf = float(inf_cfg.get("min_hand_confidence", rec_cfg.get("min_confidence", 0.5)))
    min_pred_conf = float(inf_cfg.get("min_prediction_confidence", 0.75))
    hold_ms = float(inf_cfg.get("hold_prediction_ms", 700))
    smoother = _TemporalSmoother(
        window_size=int(inf_cfg.get("smoothing_window", 9)),
        min_votes=int(inf_cfg.get("smoothing_min_votes", 4)),
    )

    cam_index = cam_cfg.get("index", 0)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))

    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam (index={cam_index})")
        sys.exit(1)

    print(f"Real-time Naruto inference started with model='{model_type}'. Press q to quit.")

    for _ in range(30):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.05)

    fps = 0.0
    ema_weight = 0.1
    prev_time = time.perf_counter()
    consecutive_failures = 0

    pred_label = "(waiting)"
    pred_conf = 0.0
    last_valid_pred_ts = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("ERROR: Too many consecutive frame read failures")
                    break
                time.sleep(0.01)
                continue
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)
            result = detector.process_frame(frame)
            frame = detector.draw_on_frame(frame)

            hands_pair = _ordered_two_hands(result, min_conf)
            now = time.perf_counter()

            status_text = "SHOW BOTH HANDS"
            status_color = (0, 0, 255)
            if hands_pair is not None:
                left_raw, right_raw = hands_pair
                feats = _features_from_two_hands(left_raw, right_raw)
                if len(feats) > 0:
                    raw_label, raw_conf = classifier.predict(feats)
                    if raw_conf < min_pred_conf:
                        elapsed_ms = (now - last_valid_pred_ts) * 1000.0
                        status_text = f"NO SIGN {raw_conf:.0%}"
                        status_color = (0, 165, 255)
                        if last_valid_pred_ts == 0 or elapsed_ms > hold_ms:
                            pred_label = "(no sign)"
                            pred_conf = raw_conf
                    else:
                        pred_label, pred_conf = smoother.update(raw_label, raw_conf)
                        last_valid_pred_ts = now
                        status_text = "TRACKING"
                        status_color = (0, 220, 0)
            else:
                elapsed_ms = (now - last_valid_pred_ts) * 1000.0
                if last_valid_pred_ts > 0 and elapsed_ms <= hold_ms:
                    status_text = f"HOLD {int(hold_ms - elapsed_ms)} ms"
                    status_color = (0, 200, 255)
                else:
                    pred_label = "(waiting)"
                    pred_conf = 0.0

            instant_fps = 1.0 / max(now - prev_time, 1e-9)
            fps = ema_weight * instant_fps + (1 - ema_weight) * fps if fps > 0 else instant_fps
            prev_time = now

            frame = _draw_hud(frame, result, fps, pred_label, pred_conf, status_text, status_color)
            cv2.imshow("Chakra Signs - Naruto Inference", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()
