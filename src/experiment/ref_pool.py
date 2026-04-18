"""Reference pool manager for few-shot voice cloning experiments.

Loads the LibriTTS-R aligned manifest and provides per-speaker reference
pools with configurable selection strategies and held-out splits.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf


@dataclass
class RefItem:
    """A single reference audio + text pair."""

    id: str
    speaker_id: str
    text: str
    path: str
    sample_rate: int
    duration: float  # seconds


class SpeakerPool:
    """Holds the ref pool and held-out eval set for one speaker."""

    def __init__(
        self,
        speaker_id: str,
        refs: list[RefItem],
        held_out: list[RefItem],
    ):
        self.speaker_id = speaker_id
        self.refs = refs
        self.held_out = held_out

    def select(
        self,
        n: int,
        strategy: str = "random",
        seed: Optional[int] = None,
    ) -> list[RefItem]:
        """Select n reference items from the pool using the given strategy.

        Strategies:
            random  – uniform random sample
            longest – pick the n longest clips
        """
        pool = list(self.refs)
        if n > len(pool):
            raise ValueError(
                f"Requested {n} refs but speaker {self.speaker_id} pool has {len(pool)}"
            )

        if strategy == "longest":
            pool.sort(key=lambda r: r.duration, reverse=True)
            return pool[:n]

        if strategy == "random":
            rng = random.Random(seed)
            return rng.sample(pool, n)

        raise ValueError(f"Unknown selection strategy: {strategy}")


def load_manifest(manifest_path: str | Path) -> list[RefItem]:
    """Load the LibriTTS-R manifest and compute durations."""
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)

    items: list[RefItem] = []
    for entry in raw:
        audio_path = entry["path"]
        try:
            info = sf.info(audio_path)
            duration = info.duration
        except Exception:
            duration = 0.0

        items.append(
            RefItem(
                id=entry["id"],
                speaker_id=entry["speaker_id"],
                text=entry["text_normalized"],
                path=audio_path,
                sample_rate=entry.get("sample_rate", 24000),
                duration=duration,
            )
        )
    return items


def build_speaker_pools(
    manifest_path: str | Path,
    held_out_per_speaker: int = 5,
    held_out_seed: int = 0,
) -> dict[str, SpeakerPool]:
    """Build per-speaker pools with deterministic held-out splits.

    The held-out clips are selected via a seeded random sample so the split
    is reproducible.  Held-out items are removed from the reference pool.
    """
    items = load_manifest(manifest_path)

    by_speaker: dict[str, list[RefItem]] = {}
    for item in items:
        by_speaker.setdefault(item.speaker_id, []).append(item)

    pools: dict[str, SpeakerPool] = {}
    for spk_id, spk_items in sorted(by_speaker.items()):
        # Sort by id for determinism before splitting
        spk_items.sort(key=lambda r: r.id)
        rng = random.Random(held_out_seed)
        if held_out_per_speaker >= len(spk_items):
            raise ValueError(
                f"Speaker {spk_id} has {len(spk_items)} clips, "
                f"cannot hold out {held_out_per_speaker}"
            )
        held_out = rng.sample(spk_items, held_out_per_speaker)
        held_out_ids = {r.id for r in held_out}
        refs = [r for r in spk_items if r.id not in held_out_ids]

        pools[spk_id] = SpeakerPool(
            speaker_id=spk_id,
            refs=refs,
            held_out=held_out,
        )

    return pools
