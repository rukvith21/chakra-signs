from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraOpenResult:
    cap: cv2.VideoCapture
    backend_name: str
    width: int
    height: int


def _candidate_backends() -> list[tuple[str, int | None]]:
    if sys.platform.startswith("win"):
        return [
            ("default", None),
            ("dshow", cv2.CAP_DSHOW),
            ("msmf", cv2.CAP_MSMF),
        ]
    return [("default", None)]


def _candidate_sizes(width: int, height: int) -> list[tuple[int, int]]:
    requested = (max(1, int(width)), max(1, int(height)))
    sizes = [requested, (1280, 720), (960, 540), (640, 480)]

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for size in sizes:
        if size not in seen:
            ordered.append(size)
            seen.add(size)
    return ordered


def _configure_capture(cap: cv2.VideoCapture, width: int, height: int) -> None:
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def _warmup_capture(
    cap: cv2.VideoCapture,
    warmup_frames: int,
    warmup_delay_s: float,
) -> bool:
    for _ in range(max(1, warmup_frames)):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
        time.sleep(warmup_delay_s)
    return False


def open_webcam(
    camera_index: int,
    width: int,
    height: int,
    warmup_frames: int = 30,
    warmup_delay_s: float = 0.05,
) -> CameraOpenResult:
    attempts: list[str] = []

    for backend_name, backend in _candidate_backends():
        for target_w, target_h in _candidate_sizes(width, height):
            cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                attempts.append(f"{backend_name}@{target_w}x{target_h}: open failed")
                cap.release()
                continue

            _configure_capture(cap, target_w, target_h)
            if _warmup_capture(cap, warmup_frames=warmup_frames, warmup_delay_s=warmup_delay_s):
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                try:
                    resolved_backend = cap.getBackendName()
                except Exception:
                    resolved_backend = backend_name
                return CameraOpenResult(
                    cap=cap,
                    backend_name=resolved_backend or backend_name,
                    width=actual_w,
                    height=actual_h,
                )

            attempts.append(f"{backend_name}@{target_w}x{target_h}: read failed")
            cap.release()

    details = "; ".join(attempts[-8:]) if attempts else "no attempts made"
    raise RuntimeError(
        f"Could not read frames from webcam index={camera_index}. Tried multiple backends/resolutions. {details}"
    )
