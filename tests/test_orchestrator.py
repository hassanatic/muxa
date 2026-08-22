import asyncio
import os

from muxa.assets import discover
from muxa.orchestrator import ledger, run_jobs
from muxa.providers.base import ProviderError
from muxa.providers.mock import MockProvider
from muxa.router import plan

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _jobs():
    return plan(discover(FIXTURES))


def test_happy_path_all_provider_route():
    results = asyncio.run(run_jobs(MockProvider(), _jobs()))
    assert results
    assert all(r.route == "provider" for r in results)
    assert all(r.text and r.tokens > 0 for r in results)


def test_injected_failure_takes_retry_route():
    jobs = _jobs()
    target = jobs[0].asset.path
    provider = MockProvider(fail_paths=frozenset({target}))
    results = asyncio.run(run_jobs(provider, jobs))
    routes = {r.path: r.route for r in results}
    assert routes[target] == "provider-retry"
    assert all(v == "provider" for k, v in routes.items() if k != target)


class AlwaysFail:
    name = "always-fail"

    async def run(self, job):
        raise ProviderError("down")


def test_fallback_never_drops_an_asset():
    jobs = _jobs()
    results = asyncio.run(run_jobs(AlwaysFail(), jobs))
    assert len(results) == len(jobs)
    assert all(r.route == "fallback" and r.tokens == 0 for r in results)
    assert all("unavailable" in r.text for r in results)


def test_ledger_accounts_for_routes_and_tokens():
    jobs = _jobs()
    target = jobs[0].asset.path
    provider = MockProvider(fail_paths=frozenset({target}))
    results = asyncio.run(run_jobs(provider, jobs))
    led = ledger(results)
    assert led["assets"] == len(jobs)
    assert led["routes"]["provider-retry"] == 1
    assert led["tokens"] == sum(r.tokens for r in results)
