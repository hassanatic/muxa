import json
import os

from muxa.cli import run as cli_run
from muxa.evalgate import score
from muxa.providers import MockProvider, ProviderError

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_eval_gate_passes_on_fixtures(monkeypatch):
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    overall, per_file, ok, stats = score()
    assert ok
    assert overall >= 0.8
    with open(os.path.join(FIXTURES, "expected.json")) as f:
        assert set(per_file) == set(json.load(f))


def test_cli_run_writes_results(tmp_path, monkeypatch):
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    assert cli_run(FIXTURES) == 0
    with open(tmp_path / "results.json") as f:
        data = json.load(f)
    assert data["provider"] == "mock"
    assert data["ledger"]["assets"] == len(data["results"]) > 0


def test_fallback_budget_fails_a_run_that_recall_alone_would_pass(monkeypatch):
    """Reliability degradation must fail on its own, not hide in a recall average.

    A provider that always fails on ONE modality still scores well overall,
    because recall is averaged across files and the other three fixtures are
    untouched. The fallback budget is what catches it.
    """
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)

    class ImageAlwaysFails(MockProvider):
        async def run(self, job):
            if job.asset.modality == "image":
                raise ProviderError("provider is down for images")
            return await super().run(job)

    overall, per_file, ok, stats = score(provider=ImageAlwaysFails())

    # The run degraded: every image came back on the fallback route.
    assert stats["fallback_rate"] > 0
    # Recall alone would NOT have caught it - the other fixtures carry the average.
    assert overall >= 0.7
    # The gate still fails, and specifically on the reliability axis.
    assert not ok
    assert stats["recall_ok"] and not stats["fallback_ok"]   # failed on reliability, not recall
    assert score(provider=ImageAlwaysFails(), max_fallback=1.0)[2] is True


def test_clean_run_reports_a_zero_fallback_rate(monkeypatch):
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    _, per_file, ok, stats = score()
    assert stats["fallback_rate"] == 0.0
    assert ok
