"""Eval gate: replay the labelled fixtures and score the run on two axes.

Each fixture file has an expectation: keywords that must appear in the text
produced for it. The gate reports recall per modality and exits non-zero
below the threshold, so a quality regression fails the build.

Recall alone is not enough. It is an average over files, so a provider that
degrades on one modality can stay above the threshold while silently serving
fallback text for every image. The README calls a spike in `fallback` routes a
monitoring signal; this gate enforces it as a budget, so reliability
regressions fail the build for their own reason rather than hiding inside a
recall average.

The same averaging problem applies to recall itself. A corpus with more text
fixtures than image ones lets image recall collapse to zero while the global
mean stays above threshold, and the gate would pass. So recall is also scored
PER MODALITY, and the weakest modality decides. Both axes exist because an
average is the wrong instrument for finding a localised failure.
"""
from __future__ import annotations

import asyncio
import json
import os

from .assets import discover
from .orchestrator import ledger, run_jobs
from .providers import from_env
from .router import plan
from .validate import validate_all

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
EXPECTED = os.path.join(FIXTURES, "expected.json")


# A fixture run uses the deterministic mock, so every asset should come back on
# the `provider` route. Anything else means the pipeline degraded; allow nothing
# by default and let a caller widen it deliberately.
MAX_FALLBACK_RATE = 0.0

# Every modality must clear the bar on its own. A global mean lets a strong
# modality carry a broken one.
MIN_PER_MODALITY_RECALL = 0.8


def fallback_rate(results) -> float:
    """Share of results that fell through to the deterministic fallback."""
    if not results:
        return 0.0
    routes = ledger(results)["routes"]
    return routes.get("fallback", 0) / len(results)


def score(threshold: float = 0.8,
          max_fallback: float = MAX_FALLBACK_RATE,
          min_per_modality: float = MIN_PER_MODALITY_RECALL,
          provider=None) -> tuple[float, dict, bool, dict]:
    """Return (overall recall, per-file recall, pass/fail, run stats).

    `per_file` stays a pure file -> recall mapping; reliability numbers travel
    in `stats` so callers can assert on either axis without one polluting the
    other.
    """
    with open(EXPECTED) as f:
        expected: dict[str, list[str]] = json.load(f)

    assets = discover(FIXTURES)
    jobs = plan(assets)
    results = validate_all(asyncio.run(run_jobs(provider or from_env(), jobs)))
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
    fb = fallback_rate(results)

    # Recall grouped by modality: the weakest modality is the one that matters.
    by_mod: dict[str, list[float]] = {}
    for name, s in per_file.items():
        r = by_name.get(name)
        if r is not None:
            by_mod.setdefault(r.modality, []).append(s)
    per_modality = {m: sum(v) / len(v) for m, v in by_mod.items()}
    weakest = min(per_modality.values()) if per_modality else 0.0

    stats = {
        "fallback_rate": fb,
        "assets": len(results),
        "per_modality": per_modality,
        "weakest_modality_recall": weakest,
        "recall_ok": overall >= threshold,
        "modality_ok": weakest >= min_per_modality,
        "fallback_ok": fb <= max_fallback,
    }
    ok = stats["recall_ok"] and stats["modality_ok"] and stats["fallback_ok"]
    return overall, per_file, ok, stats


def main() -> int:
    overall, per_file, ok, stats = score()
    for name, s in sorted(per_file.items()):
        print(f"{s:0.2f}  {name}")
    print(f"overall recall: {overall:0.2f}  ({'PASS' if stats['recall_ok'] else 'FAIL'})")
    for mod, s in sorted(stats["per_modality"].items()):
        mark = "PASS" if s >= MIN_PER_MODALITY_RECALL else "FAIL"
        print(f"  {mod:<6} recall: {s:0.2f}  ({mark}, floor {MIN_PER_MODALITY_RECALL:0.2f})")
    print(f"fallback rate:  {stats['fallback_rate']:0.2f}  "
          f"({'PASS' if stats['fallback_ok'] else 'FAIL'}, budget {MAX_FALLBACK_RATE:0.2f})")
    return 0 if ok else 1
