"""Content-addressed result cache: makes a re-run idempotent and resumable.

Why this exists: a run over a large media folder is otherwise all-or-nothing.
Interrupt it at asset 400 of 500 and the next attempt redoes, and re-pays for,
all 500. Keying finished work by CONTENT rather than by path means a second run
skips what is already done, survives a rename, and redoes exactly the assets
whose bytes actually changed.

Three rules carry most of the value:

1. **A fallback is never cached.** Fallback text is what the pipeline produces
   when a provider failed. Storing it would freeze a degraded answer forever,
   and every later run would serve that outage from disk while reporting a hit.
   Only genuine provider answers are written.
2. **The key includes the provider and a schema version.** The same bytes
   answered by a different model are a different answer, and a change to the
   result shape must invalidate everything rather than half-load.
3. **A broken cache must never break a run.** Unreadable or corrupt cache files
   are treated as empty. The cache is an optimisation; it is not the record.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

# Bump when the cached payload shape changes, so old entries can never be
# half-read into a new shape. It is part of the key, so a bump simply misses.
SCHEMA = "1"

_CHUNK = 1 << 20  # 1 MiB: media files can be large, so stream rather than slurp.


def content_digest(path: str) -> str:
    """SHA-256 of the file's bytes, streamed.

    Content rather than mtime/size on purpose: mtime changes on a copy that did
    not change the data, and size collides constantly. Hashing costs one read of
    a file we were about to send to a model anyway.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def key_for(digest: str, task: str, provider_name: str) -> str:
    """Cache key for one unit of work."""
    raw = "\0".join((SCHEMA, digest, task, provider_name))
    return hashlib.sha256(raw.encode()).hexdigest()


class ResultCache:
    """A JSON map of key -> {text, tokens}, loaded once and saved atomically."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        self.hits = 0
        self.saved_tokens = 0
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return  # missing or corrupt: rule 3, start empty rather than fail
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            self.entries = {
                k: v for k, v in data["entries"].items()
                if isinstance(v, dict) and isinstance(v.get("text"), str)
            }

    def get(self, key: str) -> dict | None:
        hit = self.entries.get(key)
        if hit is not None:
            self.hits += 1
            self.saved_tokens += int(hit.get("tokens") or 0)
        return hit

    def put(self, key: str, text: str, tokens: int) -> None:
        self.entries[key] = {"text": text, "tokens": int(tokens)}
        self._dirty = True

    def save(self) -> None:
        """Atomic write: a crash mid-save must not corrupt an existing cache."""
        if not self._dirty:
            return
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"schema": SCHEMA, "entries": self.entries}, f)
            os.replace(tmp, self.path)
            self._dirty = False
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def stats(self) -> dict:
        return {"hits": self.hits, "tokens_saved": self.saved_tokens,
                "entries": len(self.entries)}
