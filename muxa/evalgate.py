"""Eval gate: replay the labelled fixtures and score keyword recall.

Each fixture file has an expectation: keywords that must appear in the text
produced for it. The gate reports recall per modality and exits non-zero
below the threshold, so a quality regression fails the build.
"""
from __future__ import annotations

import asyncio
import json
import os

from .assets import discover
from .orchestrator import run_jobs
from .providers import from_env
from .router import plan
from .validate import validate_all

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
EXPECTED = os.path.join(FIXTURES, "expected.json")


def score(threshold: float = 0.8) -> tuple[float, dict, bool]:
    with open(EXPECTED) as f:
        expected: dict[str, list[str]] = json.load(f)

    assets = discover(FIXTURES)
    jobs = plan(assets)
    results = validate_all(asyncio.run(run_jobs(from_env(), jobs)))
    by_name = {os.path.basename(r.path): r for r in results}

    per_file: dict[str, float] = {}
    for name, keywords in expected.items():
        r = by_name.get(name)
        if r is None:
            per_file[name] = 0.0
            continue
        text = r.text.lower()
        hits = sum(1 for k in keywords if k.lower() in text)
        per_file[name] = hits / len(keywords) if keywords else 1.0

    overall = sum(per_file.values()) / len(per_file) if per_file else 0.0
    return overall, per_file, overall >= threshold


def main() -> int:
    overall, per_file, ok = score()
    for name, s in sorted(per_file.items()):
        print(f"{s:0.2f}  {name}")
    print(f"overall recall: {overall:0.2f}  ({'PASS' if ok else 'FAIL'})")
    return 0 if ok else 1
