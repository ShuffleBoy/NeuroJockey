import numpy as np
import pytest

from neurojockey.analysis.beats import fit_grid
from neurojockey.analysis.key import camelot, key_compat, shift_camelot


def _beats(bpm, n, t0=0.3, jitter=0.004, seed=0):
    rng = np.random.default_rng(seed)
    b = t0 + np.arange(n) * 60 / bpm + rng.normal(0, jitter, n)
    return np.round(b / 0.02) * 0.02  # tracker frames are 20 ms


@pytest.mark.parametrize("bpm", [122.0, 124.0, 128.0, 140.0, 174.0])
def test_fit_grid_recovers_bpm(bpm):
    beats = _beats(bpm, 600)
    downbeats = beats[2::4]
    g = fit_grid(beats, downbeats, beats[-1] + 1)
    assert g.bpm == pytest.approx(bpm, abs=0.02)
    assert not g.variable
    # bar 0 must sit on a downbeat
    assert (g.t0 - downbeats[0]) / (4 * 60 / bpm) == pytest.approx(round((g.t0 - downbeats[0]) / (4 * 60 / bpm)), abs=0.05)


def test_fit_grid_survives_spurious_and_missing_beats():
    beats = list(_beats(126.0, 500))
    beats = beats[:100] + beats[140:]  # breakdown without detected beats
    beats += [beats[50] + 0.1, beats[300] + 0.12]  # double detections
    g = fit_grid(np.sort(np.array(beats)), np.array([]), 300)
    assert g.bpm == pytest.approx(126.0, abs=0.02)


def test_camelot_mapping():
    assert camelot(0, "major") == "8B"
    assert camelot(9, "minor") == "8A"
    assert camelot(7, "major") == "9B"
    assert camelot(4, "minor") == "9A"
    assert camelot(11, "major") == "1B"


def test_key_compat_and_shift():
    assert key_compat("8A", "8A") == 1.0
    assert key_compat("8A", "9A") > 0.8
    assert key_compat("8A", "8B") > 0.8
    assert key_compat("8A", "2A") < 0.3
    assert shift_camelot("8A", 1) == "3A"  # +1 semitone = +7 on the wheel
    assert shift_camelot("8A", 2) == "10A"


def test_fit_grid_ignores_a_garbled_section():
    # Real-world failure: the tracker emitted off-grid beats for ~30 s, which used to pull
    # the global fit to 128.4 BPM with +-100 ms error.
    beats = _beats(128.0, 350, t0=0.07)
    rng = np.random.default_rng(1)
    garbled = np.sort(rng.uniform(15, 44, 70))
    beats = np.sort(np.concatenate([beats[(beats < 15) | (beats > 44)], garbled]))
    g = fit_grid(beats, np.array([]), 165)
    assert g.bpm == pytest.approx(128.0, abs=0.02)
    assert g.error_ms < 12
