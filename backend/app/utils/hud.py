from __future__ import annotations

import cv2
import numpy as np


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
