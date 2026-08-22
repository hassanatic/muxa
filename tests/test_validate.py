import pytest

from muxa.orchestrator import Result
from muxa.validate import MAX_TEXT, ValidationError, validate_all, validate_one


def _res(**kw):
    base = dict(path="/a/x.txt", modality="text", task="summarize",
                text="fine", route="provider", tokens=4)
    base.update(kw)
    return Result(**base)


def test_unknown_task_rejected():
    with pytest.raises(ValidationError):
        validate_one(_res(task="translate"))


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        validate_one(_res(text="   \n "))


def test_whitespace_normalised_and_long_text_truncated():
    r = validate_one(_res(text="a  b\n\nc"))
    assert r.text == "a b c"
    long = validate_one(_res(text="word " * 1000))
    assert len(long.text) <= MAX_TEXT + 2
    assert long.text.endswith("…")


def test_duplicate_paths_dropped_keeping_first():
    out = validate_all([_res(), _res(text="second copy")])
    assert len(out) == 1
    assert out[0].text == "fine"
