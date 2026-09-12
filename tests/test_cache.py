"""Cache behaviour, including the three failure modes it is designed against.

The interesting tests here are not the hit/miss ones. They are:
- a fallback must never be written (or an outage gets frozen on disk),
- a corrupt cache file must not break a run (the cache is not the record),
- editing a file's bytes must miss (content addressing, not path addressing).
"""
from __future__ import annotations

import asyncio
import json
import os

from muxa.assets import Asset
from muxa.cache import ResultCache, content_digest, key_for
from muxa.orchestrator import ledger, run_jobs
from muxa.providers.base import ProviderError
from muxa.router import Job


class CountingProvider:
    """Mock provider that records how many real calls it received."""
    name = "counting"

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def run(self, job: Job) -> tuple[str, int]:
        self.calls += 1
        if self.fail:
            raise ProviderError("down")
        return f"answer for {os.path.basename(job.asset.path)}", 7


def _job(tmp_path, name="notes.txt", body="hello world"):
    p = tmp_path / name
    p.write_text(body)
    asset = Asset(path=str(p), modality="text", size_bytes=len(body))
    return Job(asset=asset, task="summarize")


def test_second_run_hits_cache_and_skips_the_provider(tmp_path):
    job = _job(tmp_path)
    cache_path = str(tmp_path / "cache.json")

    first = CountingProvider()
    c1 = ResultCache(cache_path)
    r1 = asyncio.run(run_jobs(first, [job], cache=c1))
    c1.save()
    assert first.calls == 1
    assert r1[0].route == "provider"

    second = CountingProvider()
    c2 = ResultCache(cache_path)
    r2 = asyncio.run(run_jobs(second, [job], cache=c2))
    assert second.calls == 0, "a cache hit must not call the provider"
    assert r2[0].route == "cache"
    assert r2[0].text == r1[0].text
    assert r2[0].tokens == 0, "a hit spends nothing on this run"
    assert c2.stats() == {"hits": 1, "tokens_saved": 7, "entries": 1}


def test_a_fallback_is_never_cached(tmp_path):
    """The rule that matters: an outage must not be frozen on disk."""
    job = _job(tmp_path)
    cache_path = str(tmp_path / "cache.json")

    down = CountingProvider(fail=True)
    c1 = ResultCache(cache_path)
    r1 = asyncio.run(run_jobs(down, [job], cache=c1))
    c1.save()
    assert r1[0].route == "fallback"
    assert c1.entries == {}, "fallback text must not be stored"

    # Provider recovers: the next run must reach it rather than serve the outage.
    up = CountingProvider()
    c2 = ResultCache(cache_path)
    r2 = asyncio.run(run_jobs(up, [job], cache=c2))
    assert up.calls == 1
    assert r2[0].route == "provider"
    assert "answer for" in r2[0].text


def test_editing_the_file_misses_the_cache(tmp_path):
    """Content addressing: same path, different bytes, different work."""
    job = _job(tmp_path, body="first version")
    cache_path = str(tmp_path / "cache.json")

    p1 = CountingProvider()
    c1 = ResultCache(cache_path)
    asyncio.run(run_jobs(p1, [job], cache=c1))
    c1.save()

    # Rewrite the same path with different content.
    open(job.asset.path, "w").write("second version, materially different")
    p2 = CountingProvider()
    c2 = ResultCache(cache_path)
    r2 = asyncio.run(run_jobs(p2, [job], cache=c2))
    assert p2.calls == 1, "changed bytes must not hit a stale entry"
    assert r2[0].route == "provider"


def test_renaming_the_file_still_hits(tmp_path):
    """The flip side of content addressing: a rename is not new work."""
    job = _job(tmp_path, name="a.txt", body="stable content")
    cache_path = str(tmp_path / "cache.json")
    c1 = ResultCache(cache_path)
    asyncio.run(run_jobs(CountingProvider(), [job], cache=c1))
    c1.save()

    renamed = tmp_path / "b.txt"
    os.rename(job.asset.path, renamed)
    moved = Job(asset=Asset(path=str(renamed), modality="text",
                            size_bytes=len("stable content")),
                task="summarize")
    p2 = CountingProvider()
    c2 = ResultCache(cache_path)
    r2 = asyncio.run(run_jobs(p2, [moved], cache=c2))
    assert p2.calls == 0
    assert r2[0].route == "cache"


def test_corrupt_cache_file_does_not_break_the_run(tmp_path):
    """The cache is an optimisation, never the record."""
    cache_path = str(tmp_path / "cache.json")
    open(cache_path, "w").write("{ this is not json")
    job = _job(tmp_path)

    c = ResultCache(cache_path)
    assert c.entries == {}
    p = CountingProvider()
    r = asyncio.run(run_jobs(p, [job], cache=c))
    assert r[0].route == "provider"
    c.save()
    # And the corrupt file is replaced by a valid one.
    with open(cache_path) as f:
        assert isinstance(json.load(f)["entries"], dict)


def test_a_different_provider_is_a_different_key(tmp_path):
    """Same bytes answered by another model is a different answer."""
    job = _job(tmp_path)
    cache_path = str(tmp_path / "cache.json")
    c1 = ResultCache(cache_path)
    asyncio.run(run_jobs(CountingProvider(), [job], cache=c1))
    c1.save()

    other = CountingProvider()
    other.name = "someone-else"
    c2 = ResultCache(cache_path)
    r2 = asyncio.run(run_jobs(other, [job], cache=c2))
    assert other.calls == 1
    assert r2[0].route == "provider"


def test_unreadable_file_falls_through_rather_than_crashing(tmp_path):
    """A digest that cannot be computed is a normal job, not an error."""
    missing = Job(asset=Asset(path=str(tmp_path / "gone.txt"), modality="text",
                              size_bytes=0), task="summarize")
    c = ResultCache(str(tmp_path / "cache.json"))
    p = CountingProvider()
    r = asyncio.run(run_jobs(p, [missing], cache=c))
    assert p.calls == 1
    assert r[0].route == "provider"
    assert c.entries == {}, "nothing to key on, so nothing stored"


def test_no_cache_behaves_exactly_as_before(tmp_path):
    """Opt-in: passing no cache leaves the old path untouched."""
    job = _job(tmp_path)
    p = CountingProvider()
    r1 = asyncio.run(run_jobs(p, [job]))
    r2 = asyncio.run(run_jobs(p, [job]))
    assert p.calls == 2
    assert r1[0].route == r2[0].route == "provider"
    assert "cache" not in ledger(r1)


def test_ledger_separates_spend_from_savings(tmp_path):
    job = _job(tmp_path)
    cache_path = str(tmp_path / "cache.json")
    c1 = ResultCache(cache_path)
    r1 = asyncio.run(run_jobs(CountingProvider(), [job], cache=c1))
    c1.save()
    assert ledger(r1, c1)["tokens"] == 7

    c2 = ResultCache(cache_path)
    r2 = asyncio.run(run_jobs(CountingProvider(), [job], cache=c2))
    led = ledger(r2, c2)
    assert led["tokens"] == 0, "a cached run spent nothing"
    assert led["cache"]["tokens_saved"] == 7
    assert led["routes"] == {"cache": 1}


def test_digest_is_content_not_path(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("identical")
    b.write_text("identical")
    assert content_digest(str(a)) == content_digest(str(b))
    b.write_text("different")
    assert content_digest(str(a)) != content_digest(str(b))


def test_schema_and_task_are_part_of_the_key():
    d = "abc123"
    assert key_for(d, "summarize", "mock") != key_for(d, "caption", "mock")
    assert key_for(d, "summarize", "mock") != key_for(d, "summarize", "other")


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    cache_path = str(tmp_path / "sub" / "cache.json")
    c = ResultCache(cache_path)
    c.put("k", "text", 3)
    c.save()
    assert os.path.exists(cache_path)
    leftovers = [n for n in os.listdir(os.path.dirname(cache_path))
                 if n.endswith(".tmp")]
    assert leftovers == []
