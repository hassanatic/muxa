"""Route assets to tasks by modality.

The registry maps a modality to the task the pipeline should run for it.
Keeping this a plain dict makes the routing table printable and testable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .assets import Asset

ROUTES: dict[str, str] = {
    "text": "summarize",
    "image": "caption",
    "audio": "transcribe",
}


@dataclass(frozen=True)
class Job:
    asset: Asset
    task: str


def plan(assets: list[Asset]) -> list[Job]:
    """Turn discovered assets into an ordered job list."""
    return [Job(asset=a, task=ROUTES[a.modality]) for a in assets if a.modality in ROUTES]
