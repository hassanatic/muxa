"""muxa command line: `run <dir>` and `eval`."""
from __future__ import annotations

import asyncio
import json
import sys

from .assets import discover
from .orchestrator import ledger, run_jobs, to_dicts
from .providers import from_env
from .router import plan
from .validate import validate_all


def run(root: str) -> int:
    assets = discover(root)
    if not assets:
        print(f"no classifiable assets under {root}")
        return 1
    jobs = plan(assets)
    provider = from_env()
    results = validate_all(asyncio.run(run_jobs(provider, jobs)))
    out = {"provider": provider.name, "ledger": ledger(results),
           "results": to_dicts(results)}
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["ledger"], indent=2))
    print(f"{len(results)} results -> results.json")
    return 0


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        return run(sys.argv[2])
    if len(sys.argv) >= 2 and sys.argv[1] == "eval":
        from .evalgate import main as eval_main
        return eval_main()
    print("usage: python -m muxa run <dir> | python -m muxa eval")
    return 2
