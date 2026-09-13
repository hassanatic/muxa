"""muxa command line: `run <dir> [--no-cache] [--chain]` and `eval [--json]`."""
from __future__ import annotations

import asyncio
import json
import os
import sys

from .assets import discover
from .cache import ResultCache
from .orchestrator import ledger, run_jobs, to_dicts
from .providers import from_env
from .router import plan
from .validate import validate_all

CACHE_PATH = os.path.join(".muxa-cache", "results.json")


def run(root: str, use_cache: bool = True, chain: bool = False) -> int:
    assets = discover(root)
    if not assets:
        print(f"no classifiable assets under {root}")
        return 1
    jobs = plan(assets)
    provider = from_env()
    cache = ResultCache(CACHE_PATH) if use_cache else None
    results = validate_all(asyncio.run(
        run_jobs(provider, jobs, cache=cache, chain=chain)))
    if cache is not None:
        # Saved after the run, not per result: one atomic write, and an
        # interrupted run simply keeps the previous cache intact.
        cache.save()
    out = {"provider": provider.name, "ledger": ledger(results, cache),
           "results": to_dicts(results)}
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["ledger"], indent=2))
    if cache is not None and cache.hits:
        print(f"{cache.hits} cached, {cache.saved_tokens} tokens not spent again")
    print(f"{len(results)} results -> results.json")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "run" and len(argv) >= 2:
        use_cache = "--no-cache" not in argv
        chain = "--chain" in argv
        root = next((a for a in argv[1:] if not a.startswith("-")), None)
        if root is None:
            print("usage: python -m muxa run <dir> [--no-cache] [--chain]")
            return 2
        return run(root, use_cache=use_cache, chain=chain)
    if argv and argv[0] == "eval":
        from .evalgate import main as eval_main
        return eval_main(argv[1:])
    print("usage: python -m muxa run <dir> [--no-cache] [--chain] "
          "| python -m muxa eval [--json]")
    return 2
