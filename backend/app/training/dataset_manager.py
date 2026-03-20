from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import PROJECT_ROOT
from backend.app.core.schema import GestureSample

logger = logging.getLogger(__name__)

_DATA_DIR = PROJECT_ROOT / "backend" / "data"
_RAW_DIR = _DATA_DIR / "raw"
_LABELS_DIR = _DATA_DIR / "labels"
_METADATA_FILE = _LABELS_DIR / "dataset_meta.json"


class DatasetManager:
    """Persist and load gesture samples with folder organisation and versioning."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.raw_dir = Path(data_dir) / "raw" if data_dir else _RAW_DIR
        self.labels_dir = Path(data_dir) / "labels" if data_dir else _LABELS_DIR
        self.meta_path = self.labels_dir / "dataset_meta.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def _meta(self) -> dict[str, Any]:
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                return json.load(f)
        return {"version": 1, "gestures": {}, "total_samples": 0}

    def _save_meta(self, meta: dict[str, Any]) -> None:
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def save_sample(self, sample: GestureSample) -> Path:
        """Write a single sample to disk and update metadata."""
        label_dir = self.raw_dir / sample.gesture_label
        label_dir.mkdir(exist_ok=True)

        out_path = label_dir / f"{sample.sample_id}.json"
        with open(out_path, "w") as f:
            json.dump(sample.to_dict(), f, indent=2)

        meta = self._meta()
        gestures = meta.setdefault("gestures", {})
        entry = gestures.setdefault(sample.gesture_label, {"count": 0})
        entry["count"] += 1
        entry["last_updated"] = datetime.now(timezone.utc).isoformat()
        meta["total_samples"] = meta.get("total_samples", 0) + 1
        self._save_meta(meta)

        return out_path

    def save_batch(self, samples: list[GestureSample]) -> int:
        """Save many samples at once. Returns count saved."""
        for s in samples:
            self.save_sample(s)
        return len(samples)

    def load_all(self) -> list[GestureSample]:
        """Load every sample from disk."""
        samples: list[GestureSample] = []
        if not self.raw_dir.exists():
            return samples
        for json_path in sorted(self.raw_dir.rglob("*.json")):
            with open(json_path) as f:
                data = json.load(f)
            try:
                samples.append(GestureSample.from_dict(data))
            except Exception:
                logger.warning("Skipping malformed sample %s", json_path)
        return samples

    def load_label(self, label: str) -> list[GestureSample]:
        """Load samples for one gesture label."""
        label_dir = self.raw_dir / label
        samples: list[GestureSample] = []
        if not label_dir.exists():
            return samples
        for json_path in sorted(label_dir.glob("*.json")):
            with open(json_path) as f:
                data = json.load(f)
            try:
                samples.append(GestureSample.from_dict(data))
            except Exception:
                logger.warning("Skipping malformed sample %s", json_path)
        return samples

    def list_labels(self) -> list[str]:
        """Return gesture labels that have at least one sample on disk."""
        if not self.raw_dir.exists():
            return []
        return sorted(
            d.name for d in self.raw_dir.iterdir()
            if d.is_dir() and any(d.glob("*.json"))
        )

    def label_counts(self) -> dict[str, int]:
        """Return {label: sample_count} for every label."""
        return {
            label: len(list((self.raw_dir / label).glob("*.json")))
            for label in self.list_labels()
        }

    def bump_version(self) -> int:
        """Increment dataset version (e.g. after a batch of new recordings)."""
        meta = self._meta()
        meta["version"] = meta.get("version", 0) + 1
        meta["version_bumped_at"] = datetime.now(timezone.utc).isoformat()
        self._save_meta(meta)
        logger.info("Dataset version bumped to %d", meta["version"])
        return meta["version"]

    def summary(self) -> str:
        meta = self._meta()
        counts = self.label_counts()
        lines = [
            f"Dataset v{meta.get('version', '?')}  —  "
            f"{sum(counts.values())} samples across {len(counts)} gestures",
        ]
        for label, n in sorted(counts.items()):
            lines.append(f"  {label:20s}  {n:>5d} samples")
        return "\n".join(lines)
