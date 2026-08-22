from __future__ import annotations

from typing import Protocol

from ..router import Job


class ProviderError(RuntimeError):
    """Raised when a backend cannot produce a usable answer."""


class Provider(Protocol):
    name: str

    async def run(self, job: Job) -> tuple[str, int]:
        """Return (text, token_estimate) for one job. May raise ProviderError."""
        ...
