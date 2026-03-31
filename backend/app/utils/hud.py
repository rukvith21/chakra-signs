from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def text_with_bg(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_alpha: float = 0.55,
) -> None:
    """Draw *text* at *pos* with a semi-transparent dark background."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = pos
    pad = 5

    h, w = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(w, x + tw + pad)
    y2 = min(h, y + baseline + pad)

    if x1 < x2 and y1 < y2 and bg_alpha > 0:
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, bg_alpha, roi, 1 - bg_alpha, 0, roi)

    cv2.putText(frame, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)
