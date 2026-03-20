#!/usr/bin/env python3
"""Naruto hand sign recorder — captures both hands as a single gesture sample.

Usage (from project root):
    python backend/scripts/record_naruto.py

Controls (in the OpenCV window):
    1-9, 0, -, =   — select a Naruto hand sign by number
    r               — toggle recording on / off
    d               — print dataset summary to console
    q               — quit and bump dataset version
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from backend.app.core.config import DetectionConfig, load_config
from backend.app.core.schema import GestureSample
from backend.app.detection.hand_detector import DetectionResult, HandDetector
from backend.app.training.dataset_manager import DatasetManager
from backend.app.utils.hud import text_with_bg
from backend.app.utils.normalizer import normalize_landmarks

NARUTO_SIGNS: list[tuple[str, str]] = [
    ("bird",    "Tori    — pinkies interlocked, index fingers + thumbs touching"),
    ("boar",    "I       — palms flat, fingers interlocked"),
    ("dog",     "Inu     — left fist on top of right flat hand"),
    ("dragon",  "Tatsu   — left thumb over interlocked fingers"),
    ("ox",      "Ushi    — fingers interlocked horizontally"),
    ("tiger",   "Tora    — fingers interlocked, index + middle extended"),
    ("snake",   "Mi      — hands clasped, all fingers interlocked"),
    ("rat",     "Ne      — left index + middle wrapped by right hand"),
    ("horse",   "Uma     — fingers interlocked, index fingers up"),
    ("monkey",  "Saru    — right hand on left, palms together"),
    ("hare",    "U       — right middle + ring + pinky over left fist"),
    ("ram",     "Hitsuji — left index + middle up, right hand wrapping"),
]

_SIGN_KEYS = {
    ord("1"): 0, ord("2"): 1, ord("3"): 2, ord("4"): 3,
    ord("5"): 4, ord("6"): 5, ord("7"): 6, ord("8"): 7,
    ord("9"): 8, ord("0"): 9, ord("-"): 10, ord("="): 11,
}


class _NarutoRecorderState:
    def __init__(self, min_confidence: float, cooldown_ms: float) -> None:
        self.sign_index: int | None = None
        self.recording: bool = False
        self.session_id: str = uuid.uuid4().hex[:10]
        self.samples_saved: int = 0
        self.session_total: int = 0
        self.min_confidence = min_confidence
        self.cooldown_ms = cooldown_ms
        self._last_save_time: float = 0.0

    @property
    def label(self) -> str:
        if self.sign_index is not None:
            return NARUTO_SIGNS[self.sign_index][0]
        return ""

    @property
    def sign_description(self) -> str:
        if self.sign_index is not None:
            return NARUTO_SIGNS[self.sign_index][1]
        return ""

    def can_save(self) -> bool:
        if not self.recording or self.sign_index is None:
            return False
        elapsed = (time.perf_counter() - self._last_save_time) * 1000
        return elapsed >= self.cooldown_ms

    def mark_saved(self) -> None:
        self._last_save_time = time.perf_counter()
        self.samples_saved += 1
        self.session_total += 1


def _get_ordered_hands(
    result: DetectionResult,
    min_confidence: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]] | None:
    """Return (left_landmarks, right_landmarks) if 2 confident hands are found.

    Hands are assigned by MediaPipe handedness labels. If both have the same
    label, the one further left in the image is assigned "Left".
    """
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
        cx0 = np.mean([lm[0] for lm in h0.landmarks])
        cx1 = np.mean([lm[0] for lm in h1.landmarks])
        if cx0 < cx1:
            left_hand, right_hand = h0, h1
        else:
            left_hand, right_hand = h1, h0

    return left_hand.landmarks, right_hand.landmarks


def _save_two_hand_sample(
    left_raw: list[tuple[float, float, float]],
    right_raw: list[tuple[float, float, float]],
    state: _NarutoRecorderState,
    manager: DatasetManager,
) -> None:
    combined_raw = list(left_raw) + list(right_raw)
    combined_norm = normalize_landmarks(list(left_raw)) + normalize_landmarks(list(right_raw))

    sample = GestureSample(
        gesture_label=state.label,
        handedness="Both",
        normalized_landmarks=combined_norm,
        raw_landmarks=combined_raw,
        num_hands=2,
        session_id=state.session_id,
    )
    manager.save_sample(sample)
    state.mark_saved()


def _draw_hud(
    frame: np.ndarray,
    result: DetectionResult,
    fps: float,
    state: _NarutoRecorderState,
    both_hands_ok: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    text_with_bg(frame, f"FPS: {fps:.0f}", (12, 30), 0.70, (0, 255, 100))
    text_with_bg(frame, f"Hands: {result.num_hands}", (12, 64), 0.70, (0, 255, 100))

    hand_colors = {"Left": (255, 180, 0), "Right": (0, 220, 0)}
    for i, hand in enumerate(result.hands):
        color = hand_colors.get(hand.handedness, (255, 255, 255))
        y = 104 + i * 36
        text_with_bg(frame, f"{hand.handedness} | conf {hand.confidence:.0%}", (12, y), 0.62, color)

    if not both_hands_ok:
        text_with_bg(frame, "SHOW BOTH HANDS", (w // 2 - 140, h // 2), 0.9, (0, 0, 255))

    if state.recording:
        rec_color = (0, 0, 255)
        rec_text = f"REC  [{state.label}]"
    elif state.label:
        rec_color = (0, 200, 255)
        rec_text = f"READY  [{state.label}]"
    else:
        rec_color = (120, 120, 120)
        rec_text = "SELECT SIGN (1-9, 0, -, =)"

    text_with_bg(frame, rec_text, (w - 400, 30), 0.70, rec_color)

    if state.label:
        text_with_bg(frame, state.sign_description, (w - 400, 64), 0.50, (200, 200, 200))

    text_with_bg(
        frame,
        f"Saved: {state.samples_saved}  (session: {state.session_total})",
        (w - 400, 94),
        0.58,
        (200, 200, 200),
    )

    text_with_bg(frame, "r: record | 1-9,0,-,=: select sign | d: summary | q: quit", (12, h - 14), 0.50, (160, 160, 160))

    return frame


def _handle_key(key: int, state: _NarutoRecorderState, manager: DatasetManager) -> bool:
    if key == ord("q"):
        return True

    if key in _SIGN_KEYS:
        idx = _SIGN_KEYS[key]
        state.sign_index = idx
        state.recording = False
        state.samples_saved = 0
        name, desc = NARUTO_SIGNS[idx]
        print(f"  Selected: {name} — {desc}")
        return False

    if key == ord("r"):
        if state.sign_index is None:
            print("  Select a sign first (keys 1-9, 0, -, =)")
        else:
            state.recording = not state.recording
            tag = "ON" if state.recording else "OFF"
            print(f"  Recording {tag}  [{state.label}]  saved={state.samples_saved}")
    elif key == ord("d"):
        print(f"\n{manager.summary()}\n")

    return False


def main() -> None:
    cfg = load_config()
    det_cfg = cfg.get("detection", {})
    cam_cfg = cfg.get("camera", {})
    rec_cfg = cfg.get("recording", {})

    detection_config = DetectionConfig(
        num_hands=det_cfg.get("num_hands", 2),
        min_hand_detection_confidence=det_cfg.get("min_hand_detection_confidence", 0.7),
        min_hand_presence_confidence=det_cfg.get("min_hand_presence_confidence", 0.5),
        min_tracking_confidence=det_cfg.get("min_tracking_confidence", 0.5),
    )
    detector = HandDetector(detection_config)
    manager = DatasetManager()

    state = _NarutoRecorderState(
        min_confidence=rec_cfg.get("min_confidence", 0.6),
        cooldown_ms=rec_cfg.get("cooldown_ms", 150),
    )

    print("\n" + "=" * 60)
    print("  NARUTO HAND SIGN RECORDER")
    print("=" * 60)
    print("  Select a sign with the keyboard:")
    for i, (name, desc) in enumerate(NARUTO_SIGNS):
        key = "1234567890-="[i]
        print(f"    [{key}]  {name:10s}  {desc}")
    print("=" * 60)
    print(f"  Session: {state.session_id}")
    print("  r=record  d=summary  q=quit")
    print("=" * 60 + "\n")

    cam_index = cam_cfg.get("index", 0)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))

    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam (index={cam_index})")
        sys.exit(1)

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

            hands_pair = _get_ordered_hands(result, state.min_confidence)
            both_hands_ok = hands_pair is not None

            if state.can_save() and both_hands_ok:
                left_raw, right_raw = hands_pair
                _save_two_hand_sample(left_raw, right_raw, state, manager)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-9)
            fps = ema_weight * instant_fps + (1 - ema_weight) * fps if fps > 0 else instant_fps
            prev_time = now

            frame = _draw_hud(frame, result, fps, state, both_hands_ok)
            cv2.imshow("Chakra Signs - Naruto Recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and _handle_key(key, state, manager):
                break

    finally:
        if state.session_total > 0:
            ver = manager.bump_version()
            print(f"\nDataset version bumped to {ver}")
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Session {state.session_id} — {state.session_total} samples saved total.")
        print("Done.")


if __name__ == "__main__":
    main()
