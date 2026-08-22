import json
import os

from muxa.cli import run as cli_run
from muxa.evalgate import score

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_eval_gate_passes_on_fixtures(monkeypatch):
    monkeypatch.delenv("MUXA_BASE_URL", raising=False)
    overall, per_file, ok = score()
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
