"""Thin client for any OpenAI-compatible chat endpoint (incl. local Ollama).

Kept to the standard library on purpose: one POST, explicit timeouts, and a
ProviderError on anything that is not a well-formed 200.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from ..router import Job
from .base import ProviderError

_TASK_PROMPT = {
    "summarize": "Summarize this document in two sentences.",
    "caption": "Describe this image in one sentence.",
    "transcribe": "Transcribe this audio. If you cannot, describe what is known about it.",
}


class OpenAICompatProvider:
    name = "openai-compat"

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def run(self, job: Job) -> tuple[str, int]:
        content: list[dict] = [{"type": "text", "text": _TASK_PROMPT[job.task]}]
        if job.asset.modality == "image":
            with open(job.asset.path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        elif job.asset.modality == "text":
            with open(job.asset.path, "r", errors="ignore") as f:
                content.append({"type": "text", "text": f.read()[:6000]})
        else:
            content.append({"type": "text",
                            "text": f"(audio file: {job.asset.path}, {job.asset.size_bytes} bytes)"})

        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": content}]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", body,
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(str(exc)) from exc
        try:
            text = data["choices"][0]["message"]["content"]
            tokens = int(data.get("usage", {}).get("total_tokens") or max(8, len(text) // 4))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed response: {exc}") from exc
        return text, tokens
