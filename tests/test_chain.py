"""Chained tasks: a second stage that consumes the first stage's OUTPUT.

The load-bearing test here is `test_chain_refuses_to_summarise_a_fallback`.
Everything else is plumbing; that one is the reason the feature is shaped this
way rather than as a naive two-pass loop.
"""
from __future__ import annotations

import asyncio
import os

from muxa.assets import Asset
from muxa.cache import ResultCache
from muxa.orchestrator import DEGRADED_ROUTES, ledger, run_jobs
from muxa.providers.base import ProviderError
from muxa.providers.mock import MockProvider
from muxa.router import CHAINS, Job, chained_task, plan


def _audio(tmp_path, name="clip.wav", body=b"RIFF\x00\x00\x00\x00WAVEfake"):
    p = tmp_path / name
    p.write_bytes(body)
    return Asset(path=str(p), modality="audio", size_bytes=len(body))


def _text(tmp_path, name="notes.txt", body="first line here\nsecond line"):
    p = tmp_path / name
    p.write_text(body)
    return Asset(path=str(p), modality="text", size_bytes=len(body))


def test_chaining_is_off_by_default(tmp_path):
    """Opt-in: the old path must be byte-identical."""
    jobs = [Job(asset=_audio(tmp_path), task="transcribe")]
    off = asyncio.run(run_jobs(MockProvider(), jobs))
    assert len(off) == 1
    assert off[0].task == "transcribe"
    assert off[0].upstream_text is None


def test_chain_runs_a_second_stage_on_the_first_stage_text(tmp_path):
    jobs = [Job(asset=_audio(tmp_path), task="transcribe")]
    first = asyncio.run(run_jobs(MockProvider(), jobs))[0]
    out = asyncio.run(run_jobs(MockProvider(), jobs, chain=True))
    assert len(out) == 1, "a chain yields ONE result per asset, the final stage"
    r = out[0]
    assert r.task == "summarize"
    assert r.route == "provider"
    # The summary was built from the transcript, not from the file bytes.
    assert r.upstream_text == first.text
    assert first.text[:40] in r.text or "transcript" in r.text


def test_chain_refuses_to_summarise_a_fallback(tmp_path):
    """THE POINT OF THE FEATURE.

    Fallback text is an error message. Summarising it produces a confident
    summary OF an error, indistinguishable downstream from a real answer, at the
    cost of a model call. The second stage must refuse instead.
    """
    asset = _audio(tmp_path)
    jobs = [Job(asset=asset, task="transcribe")]
    # Always-failing provider: first stage exhausts its retry and falls back.
    class Down(MockProvider):
        async def run(self, job):
            raise ProviderError("down")

    out = asyncio.run(run_jobs(Down(), jobs, chain=True))
    assert len(out) == 1
    r = out[0]
    assert r.route == "blocked-upstream"
    assert r.task == "summarize"
    assert r.tokens == 0, "refusing must not cost a model call"
    assert "not attempted" in r.text
    # the failed upstream text is preserved for audit
    assert "unavailable" in (r.upstream_text or "")


def test_blocked_upstream_counts_as_degraded(tmp_path):
    """A refusal must not hide from the eval gate's reliability budget."""
    from muxa.evalgate import fallback_rate
    asset = _audio(tmp_path)

    class Down(MockProvider):
        async def run(self, job):
            raise ProviderError("down")

    out = asyncio.run(run_jobs(Down(), [Job(asset=asset, task="transcribe")], chain=True))
    assert "blocked-upstream" in DEGRADED_ROUTES
    assert fallback_rate(out) == 1.0, "a blocked chain is a degraded run, not a clean one"


def test_unchained_modalities_pass_straight_through(tmp_path):
    """Only audio declares a follow-on task; text and image must be untouched."""
    assert chained_task("audio") == "summarize"
    assert chained_task("text") is None
    assert chained_task("image") is None
    jobs = plan([_text(tmp_path)])
    out = asyncio.run(run_jobs(MockProvider(), jobs, chain=True))
    assert len(out) == 1
    assert out[0].task == "summarize"
    assert out[0].upstream_text is None, "not a chain, so no upstream"


def test_order_is_preserved_across_a_mixed_batch(tmp_path):
    a = _text(tmp_path, "a.txt", "alpha")
    b = _audio(tmp_path, "b.wav")
    c = _text(tmp_path, "c.txt", "gamma")
    jobs = [Job(asset=a, task="summarize"), Job(asset=b, task="transcribe"),
            Job(asset=c, task="summarize")]
    out = asyncio.run(run_jobs(MockProvider(), jobs, chain=True))
    assert [os.path.basename(r.path) for r in out] == ["a.txt", "b.wav", "c.txt"]
    assert out[1].task == "summarize" and out[1].upstream_text is not None
    assert out[0].upstream_text is None and out[2].upstream_text is None


def test_chained_stage_is_cached_on_the_upstream_text_not_just_the_file(tmp_path):
    """A stale-summary guard: change the transcript, the summary must re-run."""
    asset = _audio(tmp_path)
    cache_path = str(tmp_path / "c.json")

    c1 = ResultCache(cache_path)
    asyncio.run(run_jobs(MockProvider(), [Job(asset=asset, task="transcribe")],
                         cache=c1, chain=True))
    c1.save()
    assert len(c1.entries) == 2, "both stages cached"

    # Same file, same tasks: fully cached second time.
    c2 = ResultCache(cache_path)
    out = asyncio.run(run_jobs(MockProvider(), [Job(asset=asset, task="transcribe")],
                               cache=c2, chain=True))
    assert out[0].route == "cache"
    assert c2.hits == 2

    # Now force a DIFFERENT upstream text for the same file and task pair by
    # feeding the second stage directly. Its key must differ from the stored one.
    from muxa.orchestrator import _cache_key
    j_old = Job(asset=asset, task="summarize", input_text="transcript A")
    j_new = Job(asset=asset, task="summarize", input_text="transcript B")
    assert _cache_key(j_old, MockProvider()) != _cache_key(j_new, MockProvider())


def test_ledger_reports_the_chain_routes(tmp_path):
    out = asyncio.run(run_jobs(MockProvider(),
                               [Job(asset=_audio(tmp_path), task="transcribe")],
                               chain=True))
    led = ledger(out)
    assert led["assets"] == 1
    assert led["routes"] == {"provider": 1}


def test_chains_registry_is_deliberately_small(tmp_path):
    """Guard the design decision: an image caption is already a summary."""
    assert set(CHAINS) == {"audio"}
