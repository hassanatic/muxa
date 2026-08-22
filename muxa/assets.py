"""Asset discovery and modality detection.

Extensions decide the modality first; a tiny magic-byte sniff catches
mislabelled files for the common image/audio containers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

TEXT_EXT = {".txt", ".md", ".rst", ".csv", ".json", ".html"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
AUDIO_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

_MAGIC = [
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF8", "image"),
    (b"RIFF", "audio"),   # wav (RIFF....WAVE) — checked further below
    (b"ID3", "audio"),
    (b"fLaC", "audio"),
    (b"OggS", "audio"),
]


@dataclass(frozen=True)
class Asset:
    path: str
    modality: str  # "text" | "image" | "audio"
    size_bytes: int


def sniff_modality(path: str) -> str | None:
    """Return a modality from magic bytes, or None if inconclusive."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return None
    for magic, modality in _MAGIC:
        if head.startswith(magic):
            if magic == b"RIFF" and head[8:12] != b"WAVE":
                continue
            return modality
    return None


def classify(path: str) -> str | None:
    """Modality for one file: extension first, magic bytes as referee."""
    ext = os.path.splitext(path)[1].lower()
    by_ext = (
        "text" if ext in TEXT_EXT
        else "image" if ext in IMAGE_EXT
        else "audio" if ext in AUDIO_EXT
        else None
    )
    by_magic = sniff_modality(path)
    return by_magic or by_ext


def discover(root: str) -> list[Asset]:
    """Walk root and return every classifiable asset, stable order."""
    found: list[Asset] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            modality = classify(path)
            if modality is None:
                continue
            found.append(Asset(path=path, modality=modality,
                               size_bytes=os.path.getsize(path)))
    return found
