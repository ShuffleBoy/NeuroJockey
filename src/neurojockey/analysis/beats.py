"""Beat / downbeat tracking (beat_this) and constant-tempo grid fitting."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass
class Grid:
    bpm: float
    t0: float
    n_bars: int
    error_ms: float
    variable: bool


@lru_cache(maxsize=1)
def _tracker():
    from beat_this.inference import Audio2Beats

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        return Audio2Beats(checkpoint_path="final0", device=device)
    except Exception:
        return Audio2Beats(checkpoint_path="final0", device="cpu")


def track_beats(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    beats, downbeats = _tracker()(mono, sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def refine_phase(mono: np.ndarray, sr: int, bpm: float, t0: float, search: float = 0.03) -> float:
    """Nudge the grid offset (+-30 ms) so beats sit on onset peaks (tracker frames are 20 ms)."""
    import librosa

    hop = 64
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop, n_mels=64)
    fps = sr / hop
    p = 60.0 / bpm
    beat_times = t0 + p * np.arange(int((len(env) / fps - t0) / p))
    best, best_off = -1.0, 0.0
    for off in np.arange(-search, search + 1e-9, 1 / fps):
        f = np.round((beat_times + off) * fps).astype(int)
        f = f[(f >= 0) & (f < len(env))]
        s = env[f].sum()
        if s > best:
            best, best_off = s, off
    return t0 + best_off


def _robust_period(beats: np.ndarray, win: int = 32) -> float:
    """Median of local periods from windows where the tracker was consistent."""
    diffs = np.diff(beats)
    med = np.median(diffs)
    plausible = diffs[np.abs(diffs - med) < 0.15 * med]
    fallback = float(plausible.mean()) if len(plausible) else float(med)
    periods = []
    for i in range(0, len(beats) - win, win // 2):
        seg = beats[i:i + win + 1]
        k = np.concatenate([[0], np.cumsum(np.round(np.diff(seg) / fallback))])
        if k[-1] < win * 0.9 or k[-1] > win * 1.1:
            continue
        coef = np.polyfit(k, seg, 1)
        if np.abs(seg - np.polyval(coef, k)).max() < 0.04:
            periods.append(coef[0])
    return float(np.median(periods)) if len(periods) >= 2 else fallback


def _phase(beats: np.ndarray, p: float) -> float:
    """Grid offset = mode of the beat phases (robust to a minority of bad detections)."""
    ph = np.mod(beats, p)
    bins = np.linspace(0, p, max(8, int(p / 0.01)) + 1)
    hist, _ = np.histogram(ph, bins)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    centre = (bins[hist.argmax()] + bins[hist.argmax() + 1]) / 2
    near = ph[np.abs(((ph - centre + p / 2) % p) - p / 2) < 0.03]
    if len(near):
        centre = centre + np.mean(((near - centre + p / 2) % p) - p / 2)
    return float(centre)


def fit_grid(beats: np.ndarray, downbeats: np.ndarray, duration: float) -> Grid:
    """Fit a constant-tempo beat grid t = a + k*p, then find the bar phase from downbeats."""
    if len(beats) < 8:
        raise ValueError("too few beats detected")
    p = _robust_period(beats)
    a = _phase(beats, p)
    # Index every beat against the global grid (not cumulatively, so local tracker errors
    # cannot shift the rest of the track), keep inliers, refine with least squares.
    for tol in (0.06, 0.035, 0.035):
        idx = np.round((beats - a) / p)
        keep = np.abs(beats - (a + p * idx)) < tol
        if keep.sum() < 8:
            break
        A = np.stack([np.ones(keep.sum()), idx[keep]], axis=1)
        (a, p), *_ = np.linalg.lstsq(A, beats[keep], rcond=None)

    bpm = 60.0 / p
    # Most electronic tracks sit on an integer (or half) BPM; snap if close.
    for step in (1.0, 0.5):
        snapped = round(bpm / step) * step
        if abs(snapped - bpm) < 0.04:
            bpm, p = snapped, 60.0 / snapped
            idx = np.round((beats - a) / p)
            inl = np.abs(beats - (a + p * idx)) < 0.035
            a = float(np.median(beats[inl] - idx[inl] * p))
            break

    resid = beats - (a + p * np.round((beats - a) / p))
    inl = np.abs(resid) < 0.035
    error_ms = float(np.sqrt(np.mean(resid[inl] ** 2)) * 1000) if inl.any() else 999.0
    outliers = 1 - inl.mean()

    phase = 0
    if len(downbeats):
        k = np.round((downbeats - a) / p).astype(int) % 4
        phase = int(np.bincount(k, minlength=4).argmax())
    bar = 4 * p
    t0 = a + phase * p
    t0 -= np.floor((t0 + 0.02) / bar) * bar  # earliest downbeat >= ~0
    n_bars = int((duration - t0) // bar)
    return Grid(bpm=float(bpm), t0=float(t0), n_bars=n_bars, error_ms=error_ms,
                variable=bool(error_ms > 20 or outliers > 0.35))
