"""Async worker pool with retry-once and a rule-based fallback.

Design rules:
- bounded concurrency, order-independent, but results keep input order;
- a failing provider call is retried exactly once, then the deterministic
  fallback answers instead — an asset is never dropped;
- every result records which route produced it and what it cost;
- an optional content-addressed cache makes a re-run skip finished work, so a
  long run is resumable rather than all-or-nothing. It is opt-in: pass no cache
  and the pipeline behaves exactly as it did before.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import asdict, dataclass

from .cache import ResultCache, content_digest, key_for
from .providers.base import Provider, ProviderError
from .router import Job, chained_task


@dataclass(frozen=True)
class Result:
    path: str
    modality: str
    task: str
    text: str
    route: str          # "provider" | "provider-retry" | "fallback" | "cache"
                        # | "blocked-upstream"
    tokens: int
    # For a chained result, the first stage's text, kept so the intermediate is
    # auditable rather than thrown away. None for a single-stage result.
    upstream_text: str | None = None


# Routes that mean "the pipeline did not get a real answer from the provider".
# `blocked-upstream` belongs here: the second stage of a chain never ran, so
# counting it as anything other than a degradation would let a failure hide.
DEGRADED_ROUTES = frozenset({"fallback", "blocked-upstream"})


def _fallback_text(job: Job) -> str:
    stem = os.path.basename(job.asset.path)
    return (f"{job.task} unavailable for {stem}: "
            f"{job.asset.modality} file, {job.asset.size_bytes} bytes")


def _cache_key(job: Job, provider: Provider) -> str | None:
    """Key one job by its CONTENT, or None if the bytes cannot be read.

    An unreadable file is not a cache miss to be stored later; it is a job that
    must go down the normal path and be answered or fall back on its own.

    A CHAINED job's real input is `input_text`, not the file, so the upstream
    text is folded into the key. Without that, a second-stage summary keyed only
    on (file, task, provider) would be served from cache even after the upstream
    transcript changed -- a stale answer that looks like a hit. The file digest
    stays in the key too, so the entry is still tied to the asset it describes.
    """
    try:
        digest = content_digest(job.asset.path)
    except OSError:
        return None
    if job.input_text is not None:
        digest = hashlib.sha256(
            (digest + "\0" + job.input_text).encode()).hexdigest()
    return key_for(digest, job.task, getattr(provider, "name", "unknown"))


async def _run_one(provider: Provider, job: Job,
                   cache: ResultCache | None = None) -> Result:
    key = _cache_key(job, provider) if cache is not None else None
    if key is not None:
        hit = cache.get(key)
        if hit is not None:
            # tokens=0: nothing was spent on THIS run. What the hit saved is
            # accounted separately by the cache, so the run ledger stays an
            # honest record of spend rather than of work avoided.
            return Result(job.asset.path, job.asset.modality, job.task,
                          hit["text"], "cache", 0)

    for attempt, route in ((1, "provider"), (2, "provider-retry")):
        try:
            text, tokens = await provider.run(job)
            if key is not None:
                # Only a genuine provider answer is stored. Caching a fallback
                # would freeze an outage on disk forever.
                cache.put(key, text, tokens)
            return Result(job.asset.path, job.asset.modality, job.task,
                          text, route, tokens)
        except ProviderError:
            if attempt == 2:
                break
    return Result(job.asset.path, job.asset.modality, job.task,
                  _fallback_text(job), "fallback", 0)


def _chain_blocked(first: Result, task: str) -> Result:
    """Second stage refused because the first stage did not produce real output.

    THIS IS THE POINT OF THE FEATURE, not an edge case. A fallback text reads
    "transcribe unavailable for clip.wav: audio file, 40000 bytes". Feeding that
    to a summariser does not produce a degraded summary, it produces a CONFIDENT
    SUMMARY OF AN ERROR MESSAGE, which is indistinguishable downstream from a
    real one and costs a model call to manufacture. Refusing is strictly better
    than answering from garbage.
    """
    return Result(first.path, first.modality, task,
                  f"{task} not attempted: upstream {first.task} produced no usable text",
                  "blocked-upstream", 0, upstream_text=first.text)


async def run_jobs(provider: Provider, jobs: list[Job], concurrency: int = 4,
                   cache: ResultCache | None = None,
                   chain: bool = False) -> list[Result]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(job: Job) -> Result:
        async with sem:
            return await _run_one(provider, job, cache)

    first = list(await asyncio.gather(*(guarded(j) for j in jobs)))
    if not chain:
        return first

    # Second stage. Only assets whose modality declares a follow-on task, and
    # only where the first stage actually answered. Order is preserved because
    # each result keeps its index.
    out: list[Result] = []
    pending: list[tuple[int, Job, Result]] = []
    for i, r in enumerate(first):
        task = chained_task(r.modality)
        if task is None:
            out.append(r)
            continue
        out.append(r)  # placeholder, replaced below
        if r.route in DEGRADED_ROUTES:
            out[i] = _chain_blocked(r, task)
            continue
        pending.append((i, Job(asset=jobs[i].asset, task=task, input_text=r.text), r))

    async def second(idx: int, job: Job, upstream: Result) -> tuple[int, Result]:
        async with sem:
            res = await _run_one(provider, job, cache)
        return idx, Result(res.path, res.modality, res.task, res.text,
                           res.route, res.tokens, upstream_text=upstream.text)

    for idx, res in await asyncio.gather(*(second(i, j, u) for i, j, u in pending)):
        out[idx] = res
    return out


def ledger(results: list[Result], cache: ResultCache | None = None) -> dict:
    """Cost/route accounting for one run.

    `tokens` is what this run actually spent. When a cache is in play,
    `tokens_saved` reports what the hits avoided — deliberately a separate
    number, so spend is never inflated by work that did not happen.
    """
    routes: dict[str, int] = {}
    for r in results:
        routes[r.route] = routes.get(r.route, 0) + 1
    out = {
        "assets": len(results),
        "tokens": sum(r.tokens for r in results),
        "routes": routes,
    }
    if cache is not None:
        out["cache"] = cache.stats()
    return out


def to_dicts(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in results]
