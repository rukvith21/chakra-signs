#!/usr/bin/env python3
"""Phase 2 — Real-time hand landmark detection with webcam.

Usage (from project root):
    python backend/scripts/run_detection.py

Controls:
    q  — quit
    p  — print current landmarks to console
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from backend.app.core.config import AppConfig, CameraConfig, DetectionConfig, load_app_config
from backend.app.detection.hand_detector import DetectionResult, HandDetector

_HAND_COLORS = {
    "Left": (255, 180, 0),
    "Right": (0, 220, 0),
}
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-time hand landmark detection")
    parser.add_argument("--camera-index", type=int, default=None, help="Webcam index override (default from config.json)")
    parser.add_argument("--width", type=int, default=None, help="Capture width override")
    parser.add_argument("--height", type=int, default=None, help="Capture height override")
    parser.add_argument("--max-hands", type=int, choices=[1, 2], default=None, help="Maximum number of hands to detect")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=None,
        help="MediaPipe min detection confidence (0.0-1.0)",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=None,
        help="MediaPipe min tracking confidence (0.0-1.0)",
    )
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=None, help="MediaPipe model complexity")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirrored preview")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _clamp_float(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def _merge_cli_overrides(base: AppConfig, args: argparse.Namespace) -> AppConfig:
    det = base.detection
    cam = base.camera

    detection = DetectionConfig(
        max_num_hands=args.max_hands if args.max_hands is not None else det.max_num_hands,
        min_detection_confidence=(
            _clamp_float(args.min_detection_confidence)
            if args.min_detection_confidence is not None
            else det.min_detection_confidence
        ),
        min_tracking_confidence=(
            _clamp_float(args.min_tracking_confidence)
            if args.min_tracking_confidence is not None
            else det.min_tracking_confidence
        ),
        model_complexity=args.model_complexity if args.model_complexity is not None else det.model_complexity,
    )
    camera = CameraConfig(
        index=max(0, args.camera_index) if args.camera_index is not None else cam.index,
        width=max(1, args.width) if args.width is not None else cam.width,
        height=max(1, args.height) if args.height is not None else cam.height,
        mirror=False if args.no_mirror else cam.mirror,
    )
    return AppConfig(detection=detection, camera=camera)


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


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    app_config = _merge_cli_overrides(load_app_config(), args)
    detector = HandDetector(app_config.detection)

    cap = cv2.VideoCapture(app_config.camera.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, app_config.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, app_config.camera.height)

    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam (index={app_config.camera.index})")
        detector.close()
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mirror_mode = "ON" if app_config.camera.mirror else "OFF"
    print(f"Webcam opened: {actual_w}x{actual_h} (index={app_config.camera.index}, mirror={mirror_mode})")
    print("Controls:  q = quit,  p = print landmarks to console")
    print("-" * 50)

    fps = 0.0
    ema_weight = 0.1
    prev_time = time.perf_counter()
    consecutive_read_failures = 0
    max_read_failures = 10

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                logger.warning("Failed to read frame (%d/%d)", consecutive_read_failures, max_read_failures)
                if consecutive_read_failures >= max_read_failures:
                    print("ERROR: Too many failed frame reads; shutting down.")
                    break
                continue
            consecutive_read_failures = 0

            if app_config.camera.mirror:
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
            if key == ord("p"):
                _print_landmarks(result)
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nShut down cleanly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
