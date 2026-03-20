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
from backend.app.utils.hud import text_with_bg

_HAND_COLORS = {
    "Left": (255, 180, 0),
    "Right": (0, 220, 0),
}


def _draw_hud(frame: np.ndarray, result: DetectionResult, fps: float) -> np.ndarray:
    h = frame.shape[0]

    text_with_bg(frame, f"FPS: {fps:.0f}", (12, 30), 0.70, (0, 255, 100))
    text_with_bg(frame, f"Hands: {result.num_hands}", (12, 64), 0.70, (0, 255, 100))

    for i, hand in enumerate(result.hands):
        color = _HAND_COLORS.get(hand.handedness, (255, 255, 255))
        y = 104 + i * 36
        text_with_bg(frame, f"{hand.handedness} | conf {hand.confidence:.0%}", (12, y), 0.62, color)

    text_with_bg(
        frame,
        f"Detection: {result.processing_time_ms:.1f} ms",
        (12, h - 44),
        0.52,
        (180, 180, 180),
    )
    text_with_bg(frame, "q: quit | p: print landmarks", (12, h - 14), 0.50, (160, 160, 160))
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
        num_hands=det.get("num_hands", 2),
        min_hand_detection_confidence=det.get("min_hand_detection_confidence", 0.7),
        min_hand_presence_confidence=det.get("min_hand_presence_confidence", 0.5),
        min_tracking_confidence=det.get("min_tracking_confidence", 0.5),
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

    # Camera warm-up: some backends need a few frames before delivering data
    for _ in range(30):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.05)

    fps = 0.0
    ema_weight = 0.1
    prev_time = time.perf_counter()
    consecutive_failures = 0

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
