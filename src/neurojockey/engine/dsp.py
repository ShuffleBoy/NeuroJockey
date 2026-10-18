"""DSP building blocks with time-varying (per-sample) parameters."""

import numpy as np
from pedalboard import Delay, HighpassFilter, LowpassFilter, Pedalboard, Reverb
from scipy import signal

BLOCK = 256


def _lr4(x: np.ndarray, sr: int, fc: float, kind: str) -> np.ndarray:
    sos = signal.butter(2, fc, kind, fs=sr, output="sos")
    return signal.sosfilt(sos, signal.sosfilt(sos, x, axis=-1), axis=-1)


def eq3(x: np.ndarray, sr: int, low, mid, high, f1: float = 180.0, f2: float = 3000.0) -> np.ndarray:
    """DJ-style 3-band EQ (gains are linear, 0 = kill). Unity gains give the input back exactly."""
    if all(np.allclose(g, 1.0) for g in (low, mid, high)):
        return x
    lo = _lr4(x, sr, f1, "low")
    hi = _lr4(x, sr, f2, "high")
    md = x - lo - hi
    return (lo * low + md * mid + hi * high).astype(np.float32)


def _biquad(kind: str, fc: float, q: float, sr: int):
    w = 2 * np.pi * fc / sr
    alpha = np.sin(w) / (2 * q)
    cw = np.cos(w)
    if kind == "low":
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    else:
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def dj_filter(x: np.ndarray, sr: int, knob: np.ndarray) -> np.ndarray:
    """Single-knob resonant filter: knob < 0 sweeps a low-pass down, knob > 0 sweeps a high-pass up."""
    knob = np.broadcast_to(knob, x.shape[-1])
    if np.abs(knob).max() < 3e-3:
        return x
    y = x.astype(np.float64).copy()
    zi = None
    kind_prev = None
    for i0 in range(0, x.shape[-1], BLOCK):
        i1 = min(i0 + BLOCK, x.shape[-1])
        k = float(knob[(i0 + i1) // 2])
        if abs(k) < 3e-3:
            zi, kind_prev = None, None
            continue
        kind = "low" if k < 0 else "high"
        a_k = abs(k)
        fc = 20000 * (60 / 20000) ** a_k if kind == "low" else 20 * (8000 / 20) ** a_k
        b, a = _biquad(kind, min(fc, 0.45 * sr), 0.75 + 0.5 * a_k, sr)
        if zi is None or kind != kind_prev:
            zi = signal.lfilter_zi(b, a)[None, :] * y[:, i0:i0 + 1]
        y[:, i0:i1], zi = signal.lfilter(b, a, y[:, i0:i1], axis=-1, zi=zi)
        kind_prev = kind
    return y.astype(np.float32)


def echo(x: np.ndarray, sr: int, delay_s: float, feedback: float = 0.55) -> np.ndarray:
    board = Pedalboard([HighpassFilter(250), Delay(delay_seconds=delay_s, feedback=feedback, mix=1.0),
                        LowpassFilter(7000)])
    return board(x.astype(np.float32), sr)


def reverb(x: np.ndarray, sr: int, size: float = 0.9) -> np.ndarray:
    board = Pedalboard([HighpassFilter(200), Reverb(room_size=size, damping=0.35, wet_level=1.0,
                                                    dry_level=0.0, width=1.0)])
    return board(x.astype(np.float32), sr)


def noise_riser(n: int, sr: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, n)).astype(np.float32) * 0.5
    ramp = np.linspace(0, 1, n)
    x = dj_filter(x, sr, 0.15 + 0.75 * ramp)
    return (x * ramp**2).astype(np.float32)


def clap(sr: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(0.12 * sr)
    t = np.arange(n) / sr
    burst = signal.sosfilt(signal.butter(2, [900, 3000], "band", fs=sr, output="sos"), rng.standard_normal(n))
    env = np.exp(-t / 0.035) * np.minimum(t / 0.002, 1)
    return np.stack([burst * env] * 2).astype(np.float32)


def impact(sr: int) -> np.ndarray:
    n = int(1.5 * sr)
    t = np.arange(n) / sr
    sub = np.sin(2 * np.pi * np.cumsum(35 + 80 * np.exp(-t * 6)) / sr) * np.exp(-t * 2.5)
    return np.stack([sub, sub]).astype(np.float32) * 0.8


def fade_edges(x: np.ndarray, sr: int, ms: float = 4.0) -> np.ndarray:
    n = min(int(sr * ms / 1000), x.shape[-1] // 2)
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        x[:, :n] *= r
        x[:, -n:] *= r[::-1]
    return x
