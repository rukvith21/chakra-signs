from __future__ import annotations

import unittest

from backend.app.sequences import JutsuSequenceEngine, load_jutsu_patterns


class JutsuSequenceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patterns = load_jutsu_patterns(
            [
                {"name": "Fireball Jutsu", "signs": ["tiger", "ram", "snake"]},
                {"name": "Shadow Clone Jutsu", "signs": ["boar", "dog", "bird"]},
            ]
        )

    def _engine(self) -> JutsuSequenceEngine:
        return JutsuSequenceEngine(
            patterns=self.patterns,
            stable_frames=2,
            stable_confidence=0.8,
            release_frames=1,
            sequence_timeout_ms=200,
            jutsu_display_ms=500,
        )

    def test_recognizes_ordered_jutsu_and_clears_sequence(self) -> None:
        engine = self._engine()

        state = engine.update("tiger", 0.95, now=0.00)
        self.assertEqual(state.sequence, ())

        state = engine.update("tiger", 0.95, now=0.01)
        self.assertEqual(state.sequence, ("tiger",))

        state = engine.update("ram", 0.96, now=0.02)
        self.assertEqual(state.sequence, ("tiger",))

        state = engine.update("ram", 0.96, now=0.03)
        self.assertEqual(state.sequence, ("tiger", "ram"))

        state = engine.update("snake", 0.97, now=0.04)
        self.assertEqual(state.sequence, ("tiger", "ram"))

        state = engine.update("snake", 0.97, now=0.05)
        self.assertEqual(state.sequence, ())
        self.assertEqual(state.active_jutsu, "Fireball Jutsu")

    def test_prevents_duplicate_sign_adds_while_sign_is_held(self) -> None:
        engine = self._engine()

        engine.update("tiger", 0.95, now=0.00)
        state = engine.update("tiger", 0.95, now=0.01)
        self.assertEqual(state.sequence, ("tiger",))

        state = engine.update("tiger", 0.95, now=0.02)
        self.assertEqual(state.sequence, ("tiger",))

        state = engine.update("tiger", 0.95, now=0.03)
        self.assertEqual(state.sequence, ("tiger",))

    def test_resets_sequence_after_timeout(self) -> None:
        engine = self._engine()

        engine.update("tiger", 0.95, now=0.00)
        state = engine.update("tiger", 0.95, now=0.01)
        self.assertEqual(state.sequence, ("tiger",))

        state = engine.update(None, 0.0, now=0.50)
        self.assertEqual(state.sequence, ())
        self.assertEqual(state.last_event, "SEQUENCE TIMEOUT")

    def test_invalid_sequence_realigns_to_valid_suffix(self) -> None:
        engine = self._engine()

        engine.update("tiger", 0.95, now=0.00)
        state = engine.update("tiger", 0.95, now=0.01)
        self.assertEqual(state.sequence, ("tiger",))

        engine.update("boar", 0.95, now=0.02)
        state = engine.update("boar", 0.95, now=0.03)
        self.assertEqual(state.sequence, ("boar",))


if __name__ == "__main__":
    unittest.main()
