#!/usr/bin/env python3
"""Phase 3 — Interactive gesture data recorder.

Usage (from project root):
    python backend/scripts/record_gestures.py

Controls (in the OpenCV window):
    r        — toggle recording on / off
    n        — type a new gesture label (Enter to confirm, Esc to cancel)
    d        — print dataset summary to console
    q        — quit and bump dataset version
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
from backend.app.utils.camera import open_webcam
from backend.app.utils.hud import text_with_bg
from backend.app.utils.normalizer import normalize_landmarks

_HAND_COLORS = {
    "Left": (255, 180, 0),
    "Right": (0, 220, 0),
}

_ENTER = 13
_ESC = 27
_BACKSPACE_CODES = (8, 127)


class _RecorderState:
    """Mutable state for the recording session."""

    def __init__(self, min_confidence: float, cooldown_ms: float) -> None:
        self.label: str = ""
        self.recording: bool = False
        self.session_id: str = uuid.uuid4().hex[:10]
        self.samples_saved: int = 0
        self.session_total: int = 0
        self.min_confidence = min_confidence
        self.cooldown_ms = cooldown_ms
        self._last_save_time: float = 0.0

        # In-window label editor
        self.editing_label: bool = False
        self.label_buffer: str = ""

    def can_save(self) -> bool:
        if not self.recording or not self.label:
            return False
        elapsed = (time.perf_counter() - self._last_save_time) * 1000
        return elapsed >= self.cooldown_ms

    def mark_saved(self, count: int = 1) -> None:
        self._last_save_time = time.perf_counter()
        self.samples_saved += count
        self.session_total += count


def _draw_label_editor(frame: np.ndarray, buf: str) -> None:
    """Overlay a text-input bar when the user is typing a gesture label."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 40), (w, h // 2 + 40), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    prompt = f"Label: {buf}_"
    cv2.putText(
        frame, prompt,
        (40, h // 2 + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
    )
    hint = "Enter = confirm  |  Esc = cancel  |  Backspace = delete"
    cv2.putText(
        frame, hint,
        (40, h // 2 + 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA,
    )


def _draw_hud(
    frame: np.ndarray,
    result: DetectionResult,
    fps: float,
    state: _RecorderState,
) -> np.ndarray:
    h, w = frame.shape[:2]

    text_with_bg(frame, f"FPS: {fps:.0f}", (12, 30), 0.70, (0, 255, 100))
    text_with_bg(frame, f"Hands: {result.num_hands}", (12, 64), 0.70, (0, 255, 100))

    for i, hand in enumerate(result.hands):
        color = _HAND_COLORS.get(hand.handedness, (255, 255, 255))
        y = 104 + i * 36
        text_with_bg(
            frame,
            f"{hand.handedness} | conf {hand.confidence:.0%}",
            (12, y),
            0.62,
            color,
        )

    # Recording status — right side
    if state.recording:
        rec_color = (0, 0, 255)
        rec_text = f"REC  [{state.label}]"
    elif state.label:
        rec_color = (0, 200, 255)
        rec_text = f"READY  [{state.label}]"
    else:
        rec_color = (120, 120, 120)
        rec_text = "NO LABEL - press n"

    text_with_bg(frame, rec_text, (w - 380, 30), 0.70, rec_color)
    text_with_bg(
        frame,
        f"Saved: {state.samples_saved}  (session: {state.session_total})",
        (w - 380, 64),
        0.58,
        (200, 200, 200),
    )

    # Bottom bar
    text_with_bg(
        frame,
        f"Detection: {result.processing_time_ms:.1f} ms",
        (12, h - 44),
        0.52,
        (180, 180, 180),
    )
    text_with_bg(
        frame,
        "r: record | n: new label | d: summary | q: quit",
        (12, h - 14),
        0.50,
        (160, 160, 160),
    )

    if state.editing_label:
        _draw_label_editor(frame, state.label_buffer)

    return frame


def _save_frame_samples(
    result: DetectionResult,
    state: _RecorderState,
    manager: DatasetManager,
) -> int:
    """Save one sample per detected hand. Returns count saved."""
    saved = 0
    for hand in result.hands:
        if hand.confidence < state.min_confidence:
            continue
        sample = GestureSample(
            gesture_label=state.label,
            handedness=hand.handedness,
            normalized_landmarks=normalize_landmarks(hand.landmarks),
            raw_landmarks=hand.landmarks,
            num_hands=result.num_hands,
            session_id=state.session_id,
        )
        manager.save_sample(sample)
        saved += 1
    return saved


def _handle_key(key: int, state: _RecorderState, manager: DatasetManager) -> bool:
    """Process a keypress. Returns True when the main loop should exit."""
    if state.editing_label:
        if key == _ENTER:
            new_label = state.label_buffer.strip()
            if new_label:
                state.label = new_label
                state.samples_saved = 0
                print(f"  Label set to '{state.label}'")
            state.editing_label = False
            state.label_buffer = ""
        elif key == _ESC:
            state.editing_label = False
            state.label_buffer = ""
        elif key in _BACKSPACE_CODES:
            state.label_buffer = state.label_buffer[:-1]
        elif 32 <= key < 127:
            state.label_buffer += chr(key)
        return False

    if key == ord("q"):
        return True
    if key == ord("r"):
        if not state.label:
            print("  Set a label first (press n).")
        else:
            state.recording = not state.recording
            tag = "ON" if state.recording else "OFF"
            print(f"  Recording {tag}  [{state.label}]  saved={state.samples_saved}")
    elif key == ord("n"):
        state.recording = False
        state.editing_label = True
        state.label_buffer = ""
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

    state = _RecorderState(
        min_confidence=rec_cfg.get("min_confidence", 0.6),
        cooldown_ms=rec_cfg.get("cooldown_ms", 150),
    )

    cam_index = cam_cfg.get("index", 0)
    try:
        camera = open_webcam(
            camera_index=cam_index,
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    cap = camera.cap
    print(f"Webcam opened: {camera.width}x{camera.height} via {camera.backend_name}")
    print(f"Session ID: {state.session_id}")
    print("Controls:  r=record  n=new label  d=summary  q=quit")
    print("-" * 50)

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

            if state.can_save() and result.num_hands > 0:
                n = _save_frame_samples(result, state, manager)
                if n:
                    state.mark_saved(n)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-9)
            fps = ema_weight * instant_fps + (1 - ema_weight) * fps if fps > 0 else instant_fps
            prev_time = now

            frame = _draw_hud(frame, result, fps, state)
            cv2.imshow("Chakra Signs - Gesture Recorder", frame)

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
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()
