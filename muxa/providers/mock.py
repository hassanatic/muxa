"""Deterministic offline provider used by the test suite and the default CLI.

The output derives only from the file path and its first bytes, so runs are
reproducible without any network access.
"""
from __future__ import annotations

import hashlib
import os

from ..router import Job
from .base import ProviderError

_PROMPT_WORDS = {
    "summarize": "summary",
    "caption": "caption",
    "transcribe": "transcript",
}


class MockProvider:
    name = "mock"

    def __init__(self, fail_paths: frozenset[str] = frozenset()) -> None:
        # Paths listed here fail on the first call: lets tests exercise the
        # retry and fallback branches deterministically.
        self._fail_once = set(fail_paths)

    async def run(self, job: Job) -> tuple[str, int]:
        path = job.asset.path
        if path in self._fail_once:
            self._fail_once.discard(path)
            raise ProviderError(f"injected failure for {path}")
        stem = os.path.splitext(os.path.basename(path))[0]
        digest = hashlib.sha256(path.encode()).hexdigest()[:8]
        kind = _PROMPT_WORDS[job.task]
        if job.task == "summarize":
            with open(path, "r", errors="ignore") as f:
                first_line = (f.readline() or "").strip()[:80]
            text = f"{kind} of {stem}: {first_line or 'empty file'}"
        else:
            text = f"{kind} of {stem} ({job.asset.modality}, {job.asset.size_bytes} bytes, {digest})"
        return text, max(8, len(text) // 4)
