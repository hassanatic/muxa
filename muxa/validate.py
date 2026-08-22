"""Validate model output into a closed result shape.

The orchestrator returns free text from whatever backend answered; this module
is the boundary that turns it into something downstream code can trust:
non-empty, sane length, tagged with a known task, deduplicated per path.
"""
from __future__ import annotations

from .orchestrator import Result

KNOWN_TASKS = {"summarize", "caption", "transcribe"}
MAX_TEXT = 2000


class ValidationError(ValueError):
    pass


def validate_one(result: Result) -> Result:
    if result.task not in KNOWN_TASKS:
        raise ValidationError(f"unknown task {result.task!r} for {result.path}")
    text = " ".join(result.text.split())
    if not text:
        raise ValidationError(f"empty text for {result.path}")
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT].rsplit(" ", 1)[0] + " …"
    if result.tokens < 0:
        raise ValidationError(f"negative token count for {result.path}")
    if text != result.text:
        result = Result(result.path, result.modality, result.task,
                        text, result.route, result.tokens)
    return result


def validate_all(results: list[Result]) -> list[Result]:
    """Validate every result and drop exact-path duplicates, keeping the first."""
    seen: set[str] = set()
    clean: list[Result] = []
    for r in results:
        if r.path in seen:
            continue
        seen.add(r.path)
        clean.append(validate_one(r))
    return clean
