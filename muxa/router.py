"""Route assets to tasks by modality, optionally as a short chain.

The registry maps a modality to the task the pipeline should run for it.
Keeping this a plain dict makes the routing table printable and testable.

CHAINS exist because some modalities have no single useful answer. Transcribing
an hour of audio produces a wall of text that nobody reads; what a caller
actually wants is a summary OF that transcript. That is two model calls where
the second consumes the FIRST ONE'S OUTPUT rather than the file, which is a
different shape from everything else here and the reason this is a chain rather
than another entry in ROUTES.

It is OPT-IN. With chaining off, `plan()` returns exactly what it always did and
every downstream contract (one result per asset, the eval gate's per-modality
recall, validate's path dedup) is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

from .assets import Asset

ROUTES: dict[str, str] = {
    "text": "summarize",
    "image": "caption",
    "audio": "transcribe",
}

# modality -> the follow-on task that consumes the first task's OUTPUT.
# Only audio earns one: a transcript is raw material, not an answer. An image
# caption is already a summary, and summarising a summary adds nothing but a
# second chance to hallucinate.
CHAINS: dict[str, str] = {
    "audio": "summarize",
}


@dataclass(frozen=True)
class Job:
    asset: Asset
    task: str
    # When set, the provider must answer from THIS TEXT rather than from the
    # file on disk. That is what makes a chain a chain.
    input_text: str | None = None


def plan(assets: list[Asset]) -> list[Job]:
    """Turn discovered assets into an ordered job list (first stage only)."""
    return [Job(asset=a, task=ROUTES[a.modality]) for a in assets if a.modality in ROUTES]


def chained_task(modality: str) -> str | None:
    """The follow-on task for a modality, or None if it has no second stage."""
    return CHAINS.get(modality)
