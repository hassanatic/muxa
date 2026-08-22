import os

from muxa.assets import classify, discover
from muxa.router import ROUTES, plan

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_classify_by_extension(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello")
    assert classify(str(p)) == "text"


def test_magic_overrides_wrong_extension(tmp_path):
    p = tmp_path / "photo.txt"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    assert classify(str(p)) == "image"


def test_riff_without_wave_is_not_audio(tmp_path):
    p = tmp_path / "clip.bin"
    p.write_bytes(b"RIFF\x00\x00\x00\x00AVI LIST")
    assert classify(str(p)) is None


def test_discover_fixture_modalities():
    by_name = {os.path.basename(a.path): a.modality for a in discover(FIXTURES)}
    assert by_name["notes.txt"] == "text"
    assert by_name["logo.png"] == "image"
    assert by_name["clip.wav"] == "audio"


def test_plan_maps_modality_to_task():
    jobs = plan(discover(FIXTURES))
    assert jobs
    for job in jobs:
        assert job.task == ROUTES[job.asset.modality]
