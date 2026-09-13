import json
import os

from muxa.cli import run as cli_run
from muxa import evalgate
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


def test_per_modality_floor_catches_a_collapse_the_global_mean_hides(monkeypatch):
    """A weak modality must fail on its own, not be carried by stronger ones.

    This provider answers image jobs successfully but with useless text, so
    nothing routes to fallback and the reliability axis stays clean. The fixture
    corpus is deliberately imbalanced (4 text, 1 image, 1 audio), so one dead
    modality leaves the global mean at 0.833 - still above the 0.8 threshold.
    The global axis therefore PASSES and only the per-modality floor sees it.

    That imbalance is the point: with a balanced 4-file corpus the mean drops to
    0.75 and the global threshold catches the collapse by luck, which would make
    this feature look unnecessary.
    """
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)

    class ImageAnswersUselessly(MockProvider):
        async def run(self, job):
            if job.asset.modality == "image":
                return "ok", 8          # succeeds, but contains no expected keyword
            return await super().run(job)

    overall, per_file, ok, stats = score(provider=ImageAnswersUselessly())

    assert stats["fallback_rate"] == 0.0          # reliability axis is clean
    assert stats["recall_ok"]                     # the GLOBAL mean still passes (0.833)
    assert overall >= 0.8
    assert stats["per_modality"]["image"] == 0.0  # the collapse is real
    assert stats["per_modality"]["text"] == 1.0   # and localised
    assert not stats["modality_ok"]               # only the per-modality floor catches it
    assert not ok

    # Relaxing only the per-modality floor makes the same run pass, which pins
    # WHICH axis rejected it.
    assert score(provider=ImageAnswersUselessly(), min_per_modality=0.0)[3]["modality_ok"] is True


def test_clean_run_reports_every_modality(monkeypatch):
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    _, _, ok, stats = score()
    assert ok
    assert set(stats["per_modality"]) == {"text", "image", "audio"}
    assert stats["weakest_modality_recall"] == 1.0


def test_eval_json_emits_only_a_parseable_record(monkeypatch, capsys):
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    rc = evalgate.main(["--json"])
    assert rc == 0
    # The whole of stdout must parse. A header line printed alongside the record
    # would make `muxa eval --json > gate.json` produce an unreadable file.
    record = json.loads(capsys.readouterr().out)
    assert record["ok"] is True
    assert record["version"] == evalgate.REPORT_VERSION
    assert set(record["checks"]) == {"recall_ok", "modality_ok", "fallback_ok"}
    assert set(record["per_modality_recall"]) == {"audio", "image", "text"}


def test_record_carries_the_thresholds_it_was_judged_against(monkeypatch):
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    overall, per_file, ok, stats = evalgate.score()
    record = evalgate.report(overall, per_file, ok, stats)
    # An archived measurement without its bar cannot be re-checked later.
    assert record["thresholds"] == {
        "overall_recall": evalgate.MIN_OVERALL_RECALL,
        "per_modality_recall": evalgate.MIN_PER_MODALITY_RECALL,
        "max_fallback_rate": evalgate.MAX_FALLBACK_RATE,
    }


def test_printed_floor_tracks_the_applied_floor(monkeypatch, capsys):
    """The displayed PASS/FAIL and the decision must share one source.

    Two defects at once if they do not. `main` used to compare each modality
    against the module constant while the verdict came from `score`, so the two
    could disagree; and `score` took its thresholds as DEFAULT ARGUMENTS, which
    Python binds at definition time, so raising the constant here would not have
    reached the gate at all. Raising it above 1.0 must now flip the result and
    change the printed floor together.
    """
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    monkeypatch.setattr(evalgate, "MIN_PER_MODALITY_RECALL", 1.5)
    rc = evalgate.main([])
    out = capsys.readouterr().out
    assert rc == 1, "an unreachable floor must fail the gate"
    assert "floor 1.50" in out, "the printed floor must be the applied one"
    assert "FAIL" in out


def test_json_exit_code_still_reports_failure(monkeypatch, capsys):
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    monkeypatch.setattr(evalgate, "MAX_FALLBACK_RATE", -1.0)
    rc = evalgate.main(["--json"])
    record = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert record["ok"] is False
    assert record["checks"]["fallback_ok"] is False
    assert record["thresholds"]["max_fallback_rate"] == -1.0
