from __future__ import annotations

import numpy as np


def normalize_landmarks(
    raw: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Normalize 21 hand landmarks to be position- and scale-invariant.

    1. Translate so the wrist (landmark 0) sits at the origin.
    2. Scale so the distance from wrist to middle-finger MCP (landmark 9)
       equals 1.0.  If that distance is near-zero the raw values are returned
       unchanged (degenerate detection).
    """
    arr = np.array(raw, dtype=np.float64)  # (21, 3)
    wrist = arr[0]
    arr = arr - wrist

    ref_dist = float(np.linalg.norm(arr[9]))
    if ref_dist < 1e-8:
        return raw

    arr = arr / ref_dist
    return [(float(x), float(y), float(z)) for x, y, z in arr]
