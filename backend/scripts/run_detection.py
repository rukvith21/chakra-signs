#!/usr/bin/env python3
"""Phase 2 — Real-time hand landmark detection with webcam.

Usage (from project root):
    python backend/scripts/run_detection.py

Controls:
    q  — quit
    p  — print current landmarks to console
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from backend.app.core.config import DetectionConfig, load_config
from backend.app.detection.hand_detector import DetectionResult, HandDetector

_HAND_COLORS = {
    "Left": (255, 180, 0),
    "Right": (0, 220, 0),
}


def _text_with_bg(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_alpha: float = 0.55,
) -> None:
    """Draw *text* at *pos* with a semi-transparent dark background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    pad = 5
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, result: DetectionResult, fps: float) -> np.ndarray:
    h = frame.shape[0]

    _text_with_bg(frame, f"FPS: {fps:.0f}", (12, 30), 0.70, (0, 255, 100))
    _text_with_bg(frame, f"Hands: {result.num_hands}", (12, 64), 0.70, (0, 255, 100))

    for i, hand in enumerate(result.hands):
        color = _HAND_COLORS.get(hand.handedness, (255, 255, 255))
        y = 104 + i * 36
        _text_with_bg(frame, f"{hand.handedness} | conf {hand.confidence:.0%}", (12, y), 0.62, color)

    _text_with_bg(
        frame,
        f"Detection: {result.processing_time_ms:.1f} ms",
        (12, h - 44),
        0.52,
        (180, 180, 180),
    )
    _text_with_bg(frame, "q: quit | p: print landmarks", (12, h - 14), 0.50, (160, 160, 160))
    return frame


def _print_landmarks(result: DetectionResult) -> None:
    if result.num_hands == 0:
        print("[No hands detected]")
        return
    for hand in result.hands:
        print(f"\n--- {hand.handedness} hand (confidence: {hand.confidence:.2%}) ---")
        for idx, (x, y, z) in enumerate(hand.landmarks):
            print(f"  [{idx:2d}] x={x:.4f}  y={y:.4f}  z={z:.6f}")


def main() -> None:
    cfg = load_config()
    det = cfg.get("detection", {})
    cam = cfg.get("camera", {})

    detection_config = DetectionConfig(
        max_num_hands=det.get("max_num_hands", 2),
        min_detection_confidence=det.get("min_detection_confidence", 0.7),
        min_tracking_confidence=det.get("min_tracking_confidence", 0.5),
        model_complexity=det.get("model_complexity", 1),
    )
    detector = HandDetector(detection_config)

    cam_index = cam.get("index", 0)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 720))

    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam (index={cam_index})")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: {actual_w}x{actual_h}")
    print("Controls:  q = quit,  p = print landmarks to console")
    print("-" * 50)

    fps = 0.0
    ema_weight = 0.1
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Failed to read frame")
                break

            frame = cv2.flip(frame, 1)
            result = detector.process_frame(frame)
            frame = detector.draw_on_frame(frame)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-9)
            fps = ema_weight * instant_fps + (1 - ema_weight) * fps if fps > 0 else instant_fps
            prev_time = now

            frame = _draw_hud(frame, result, fps)
            cv2.imshow("Chakra Signs - Hand Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                _print_landmarks(result)
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nShut down cleanly.")


if __name__ == "__main__":
    main()
