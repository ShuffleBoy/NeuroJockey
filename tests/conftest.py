import pytest

from neurojockey.analysis.model import Highlight, Segment, TrackAnalysis
from neurojockey.audio import save
from neurojockey.synth import generate

FORM = [("intro", 8), ("build", 4), ("drop", 16), ("break", 8), ("drop", 16), ("outro", 8)]


def fake_analysis(path: str, bpm: float, camelot: str, energy: float, tid: str) -> TrackAnalysis:
    segs, bar = [], 0
    for label, n in FORM:
        segs.append(Segment(bar, bar + n, label, 0.9 if label == "drop" else 0.3))
        bar += n
    return TrackAnalysis(
        path=path, track_id=tid, title=tid, duration=bar * 240 / bpm + 1.25, bpm=bpm, t0=0.25,
        n_bars=bar, variable_tempo=False, grid_error_ms=1.0, key="", key_pc=0, key_mode="minor",
        camelot=camelot, key_confidence=1.0, lufs=-12.0, energy_level=energy,
        bar_energy=[0.5] * bar, bar_vocal=None, segments=segs,
        highlights=[Highlight(12, 28, 1.0, "drop")], has_stems=False, version=1,
    )


@pytest.fixture(scope="session")
def demo_tracks(tmp_path_factory):
    d = tmp_path_factory.mktemp("tracks")
    specs = [(124, 9, "minor", "8A", 0.3), (126, 4, "minor", "9A", 0.5), (128, 2, "minor", "7A", 0.7), (150, 7, "minor", "6A", 0.9)]
    out = []
    for i, (bpm, pc, mode, cam, e) in enumerate(specs):
        p = d / f"t{i}.wav"
        save(p, generate(bpm, pc, mode, form=FORM, seed=i))
        out.append(fake_analysis(str(p), bpm, cam, e, f"t{i}"))
    return out
