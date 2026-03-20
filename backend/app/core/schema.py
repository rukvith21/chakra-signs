from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GestureSample:
    """One recorded snapshot of a single hand performing a gesture."""

    gesture_label: str
    handedness: str
    normalized_landmarks: list[tuple[float, float, float]]
    raw_landmarks: list[tuple[float, float, float]]
    num_hands: int = 1
    sample_id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now_iso)
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GestureSample:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
