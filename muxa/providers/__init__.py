"""Provider backends: a call is (task, asset) -> raw text."""
from __future__ import annotations

import os

from .base import Provider, ProviderError
from .mock import MockProvider


def from_env() -> Provider:
    """Pick a provider from the environment; the mock is the default."""
    base_url = os.environ.get("MUXA_BASE_URL")
    if base_url:
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            base_url=base_url,
            model=os.environ.get("MUXA_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("MUXA_API_KEY", "unused"),
        )
    return MockProvider()


__all__ = ["Provider", "ProviderError", "MockProvider", "from_env"]
