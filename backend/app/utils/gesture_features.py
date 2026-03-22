from __future__ import annotations

import numpy as np

from backend.app.core.schema import GestureSample
from backend.app.utils.normalizer import normalize_landmarks


def flatten_landmarks(landmarks: list[tuple[float, float, float]]) -> list[float]:
    features: list[float] = []
    for x, y, z in landmarks:
        features.extend([x, y, z])
    return features


def _normalize_two_hand_global(
    left_raw: list[tuple[float, float, float]],
    right_raw: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Normalize both hands together so relative hand layout is preserved."""
    left_arr = np.array(left_raw, dtype=np.float64)
    right_arr = np.array(right_raw, dtype=np.float64)
    combined = np.vstack([left_arr, right_arr])

    origin = (left_arr[0] + right_arr[0]) / 2.0
    left_scale = float(np.linalg.norm(left_arr[9] - left_arr[0]))
    right_scale = float(np.linalg.norm(right_arr[9] - right_arr[0]))
    scale = (left_scale + right_scale) / 2.0
    if scale < 1e-8:
        return list(left_raw) + list(right_raw)

    combined = (combined - origin) / scale
    return [(float(x), float(y), float(z)) for x, y, z in combined]


def build_feature_vector(
    raw_landmarks: list[tuple[float, float, float]],
) -> list[float]:
    """Build a model feature vector from raw landmarks.

    Single-hand samples keep the existing wrist-relative normalization.
    Two-hand samples include both:
    - per-hand normalized shape
    - globally normalized two-hand geometry
    """
    if len(raw_landmarks) == 21:
        return flatten_landmarks(normalize_landmarks(list(raw_landmarks)))

    if len(raw_landmarks) == 42:
        left_raw = list(raw_landmarks[:21])
        right_raw = list(raw_landmarks[21:])
        local_shape = normalize_landmarks(left_raw) + normalize_landmarks(right_raw)
        global_layout = _normalize_two_hand_global(left_raw, right_raw)
        return flatten_landmarks(local_shape) + flatten_landmarks(global_layout)

    raise ValueError(
        f"Unsupported landmark count: {len(raw_landmarks)}. "
        "Expected 21 for single-hand or 42 for two-hand samples."
    )


def sample_feature_vector(sample: GestureSample) -> list[float]:
    raw = sample.raw_landmarks or sample.normalized_landmarks
    return build_feature_vector(raw)
