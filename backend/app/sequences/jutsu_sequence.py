from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


_DEFAULT_PATTERNS = (
    ("Fireball Jutsu", ("tiger", "ram", "snake")),
    ("Shadow Clone Jutsu", ("boar", "dog", "bird")),
)


def normalize_sign_name(sign: str) -> str:
    cleaned = str(sign).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class JutsuPattern:
    name: str
    signs: tuple[str, ...]


@dataclass(frozen=True)
class SequenceState:
    sequence: tuple[str, ...]
    active_jutsu: str | None
    candidate_sign: str | None
    candidate_frames: int
    locked_sign: str | None
    last_event: str


def load_jutsu_patterns(raw_patterns: Any = None) -> tuple[JutsuPattern, ...]:
    if raw_patterns is None:
        return tuple(JutsuPattern(name=name, signs=signs) for name, signs in _DEFAULT_PATTERNS)

    entries: list[dict[str, Any]] = []
    if isinstance(raw_patterns, dict):
        for name, signs in raw_patterns.items():
            entries.append({"name": name, "signs": signs})
    elif isinstance(raw_patterns, (list, tuple)):
        for entry in raw_patterns:
            if isinstance(entry, dict):
                entries.append(entry)

    patterns: list[JutsuPattern] = []
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        signs_raw = entry.get("signs", [])
        if not name or not isinstance(signs_raw, (list, tuple)):
            continue

        signs = tuple(
            normalized
            for raw_sign in signs_raw
            if (normalized := normalize_sign_name(str(raw_sign)))
        )
        if signs:
            patterns.append(JutsuPattern(name=name, signs=signs))

    if not patterns:
        return tuple(JutsuPattern(name=name, signs=signs) for name, signs in _DEFAULT_PATTERNS)
    return tuple(patterns)


class JutsuSequenceEngine:
    """Build a stable sign sequence and match it against jutsu patterns."""

    def __init__(
        self,
        patterns: tuple[JutsuPattern, ...] | list[JutsuPattern],
        stable_frames: int = 4,
        stable_confidence: float = 0.8,
        release_frames: int = 3,
        sequence_timeout_ms: float = 3000.0,
        jutsu_display_ms: float = 2200.0,
    ) -> None:
        self.patterns = tuple(patterns)
        self.stable_frames = max(1, int(stable_frames))
        self.stable_confidence = float(stable_confidence)
        self.release_frames = max(1, int(release_frames))
        self.sequence_timeout_s = max(0.0, float(sequence_timeout_ms)) / 1000.0
        self.jutsu_display_s = max(0.0, float(jutsu_display_ms)) / 1000.0

        self._sequence: list[str] = []
        self._candidate_sign: str | None = None
        self._candidate_frames = 0
        self._locked_sign: str | None = None
        self._release_count = 0
        self._last_step_ts = 0.0
        self._active_jutsu: str | None = None
        self._active_jutsu_ts = 0.0
        self._last_event = "READY"

    def update(self, label: str | None, confidence: float, now: float | None = None) -> SequenceState:
        timestamp = time.perf_counter() if now is None else now
        self._expire_sequence(timestamp)
        self._expire_jutsu(timestamp)

        normalized_label = normalize_sign_name(label) if label else ""
        if not normalized_label or confidence < self.stable_confidence:
            self._handle_gap()
            return self.state

        self._release_count = 0

        if normalized_label == self._locked_sign:
            self._clear_candidate()
            self._last_event = f"HOLDING {normalized_label.upper()}"
            return self.state

        if normalized_label != self._candidate_sign:
            self._candidate_sign = normalized_label
            self._candidate_frames = 1
            self._last_event = f"STABILIZING {normalized_label.upper()} 1/{self.stable_frames}"
            return self.state

        self._candidate_frames += 1
        if self._candidate_frames < self.stable_frames:
            self._last_event = (
                f"STABILIZING {normalized_label.upper()} "
                f"{self._candidate_frames}/{self.stable_frames}"
            )
            return self.state

        self._append_sign(normalized_label, timestamp)
        return self.state

    @property
    def state(self) -> SequenceState:
        return SequenceState(
            sequence=tuple(self._sequence),
            active_jutsu=self._active_jutsu,
            candidate_sign=self._candidate_sign,
            candidate_frames=self._candidate_frames,
            locked_sign=self._locked_sign,
            last_event=self._last_event,
        )

    def _handle_gap(self) -> None:
        self._clear_candidate()
        self._release_count += 1
        released_now = False
        if self._release_count >= self.release_frames:
            released_now = self._locked_sign is not None
            self._locked_sign = None
        if (
            released_now
            and not self._sequence
            and self._active_jutsu is None
            and self._last_event not in {"SEQUENCE TIMEOUT", "RESET INVALID SEQUENCE"}
        ):
            self._last_event = "READY"

    def _append_sign(self, sign: str, now: float) -> None:
        self._sequence.append(sign)
        self._locked_sign = sign
        self._last_step_ts = now
        self._clear_candidate()

        matched = self._match_exact(tuple(self._sequence))
        if matched is not None:
            self._active_jutsu = matched.name
            self._active_jutsu_ts = now
            self._sequence.clear()
            self._last_event = f"JUTSU {matched.name}"
            return

        self._sequence = self._reduce_to_valid_suffix(self._sequence)
        if self._sequence:
            self._last_event = f"ADDED {sign.upper()}"
        else:
            self._last_event = "RESET INVALID SEQUENCE"

    def _expire_sequence(self, now: float) -> None:
        if not self._sequence or self.sequence_timeout_s <= 0 or self._last_step_ts <= 0:
            return
        if now - self._last_step_ts > self.sequence_timeout_s:
            self._sequence.clear()
            self._last_step_ts = 0.0
            self._last_event = "SEQUENCE TIMEOUT"

    def _expire_jutsu(self, now: float) -> None:
        if self._active_jutsu is None or self.jutsu_display_s <= 0:
            return
        if now - self._active_jutsu_ts > self.jutsu_display_s:
            self._active_jutsu = None
            if not self._sequence:
                self._last_event = "READY"

    def _clear_candidate(self) -> None:
        self._candidate_sign = None
        self._candidate_frames = 0

    def _match_exact(self, signs: tuple[str, ...]) -> JutsuPattern | None:
        for pattern in self.patterns:
            if pattern.signs == signs:
                return pattern
        return None

    def _is_prefix(self, signs: tuple[str, ...]) -> bool:
        if not signs:
            return False
        for pattern in self.patterns:
            if pattern.signs[: len(signs)] == signs:
                return True
        return False

    def _reduce_to_valid_suffix(self, signs: list[str]) -> list[str]:
        if not signs:
            return []

        for start in range(len(signs)):
            suffix = tuple(signs[start:])
            if self._is_prefix(suffix):
                return list(suffix)
        return []
