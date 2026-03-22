from __future__ import annotations

import unittest

import numpy as np

from backend.app.utils.gesture_features import build_feature_vector


def _make_hand(base_x: float, base_y: float, base_z: float) -> list[tuple[float, float, float]]:
    return [
        (base_x + i * 0.01, base_y + i * 0.02, base_z + i * 0.005)
        for i in range(21)
    ]


class GestureFeatureTests(unittest.TestCase):
    def test_single_hand_feature_size_is_63(self) -> None:
        hand = _make_hand(0.0, 0.0, 0.0)
        features = build_feature_vector(hand)
        self.assertEqual(len(features), 63)

    def test_two_hand_feature_size_is_252(self) -> None:
        left = _make_hand(0.0, 0.0, 0.0)
        right = _make_hand(1.0, 0.0, 0.0)
        features = build_feature_vector(left + right)
        self.assertEqual(len(features), 252)

    def test_two_hand_features_preserve_relative_layout(self) -> None:
        left = _make_hand(0.0, 0.0, 0.0)
        right_near = _make_hand(1.0, 0.0, 0.0)
        right_far = _make_hand(3.0, 0.0, 0.0)

        near_features = build_feature_vector(left + right_near)
        far_features = build_feature_vector(left + right_far)

        self.assertNotEqual(near_features, far_features)
        self.assertTrue(np.allclose(near_features[:126], far_features[:126]))
        self.assertFalse(np.allclose(near_features[126:], far_features[126:]))


if __name__ == "__main__":
    unittest.main()
