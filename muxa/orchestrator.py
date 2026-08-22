"""Async worker pool with retry-once and a rule-based fallback.

Design rules:
- bounded concurrency, order-independent, but results keep input order;
- a failing provider call is retried exactly once, then the deterministic
  fallback answers instead — an asset is never dropped;
- every result records which route produced it and what it cost.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass

from .providers.base import Provider, ProviderError
from .router import Job


@dataclass(frozen=True)
class Result:
    path: str
    modality: str
    task: str
    text: str
    route: str          # "provider" | "provider-retry" | "fallback"
    tokens: int


def _fallback_text(job: Job) -> str:
    stem = os.path.basename(job.asset.path)
    return (f"{job.task} unavailable for {stem}: "
            f"{job.asset.modality} file, {job.asset.size_bytes} bytes")


async def _run_one(provider: Provider, job: Job) -> Result:
    for attempt, route in ((1, "provider"), (2, "provider-retry")):
        try:
            text, tokens = await provider.run(job)
            return Result(job.asset.path, job.asset.modality, job.task,
                          text, route, tokens)
        except ProviderError:
            if attempt == 2:
                break
    return Result(job.asset.path, job.asset.modality, job.task,
                  _fallback_text(job), "fallback", 0)


async def run_jobs(provider: Provider, jobs: list[Job], concurrency: int = 4) -> list[Result]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(job: Job) -> Result:
        async with sem:
            return await _run_one(provider, job)

    return list(await asyncio.gather(*(guarded(j) for j in jobs)))


def ledger(results: list[Result]) -> dict:
    """Cost/route accounting for one run."""
    routes: dict[str, int] = {}
    for r in results:
        routes[r.route] = routes.get(r.route, 0) + 1
    return {
        "assets": len(results),
        "tokens": sum(r.tokens for r in results),
        "routes": routes,
    }


def to_dicts(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in results]
