"""Synthetic EDM-ish track generator with known BPM / key / structure (for demos and tests)."""

import numpy as np
from scipy import signal

from .config import SR

DEFAULT_FORM = [("intro", 16), ("build", 8), ("drop", 32), ("break", 16), ("build", 8), ("drop", 32), ("outro", 16)]

_LAYERS = {
    "intro": {"kick", "hat"},
    "build": {"hat", "pad", "roll", "riser"},
    "drop": {"kick", "hat", "clap", "bass", "pad", "vocal"},
    "break": {"pad", "vocal"},
    "outro": {"kick", "hat"},
}


def _midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def _add(buf: np.ndarray, x: np.ndarray, at: int, gain: float = 1.0) -> None:
    end = min(len(buf), at + len(x))
    if end > at >= 0:
        buf[at:end] += gain * x[: end - at]


def _saw(freq: float, n: int, detune: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    return 2 * ((t * freq * (1 + detune)) % 1.0) - 1


def generate(bpm: float = 124, root_pc: int = 9, mode: str = "minor", form=None, seed: int = 0) -> np.ndarray:
    form = form or DEFAULT_FORM
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar_n = 4 * beat * SR
    total_bars = sum(b for _, b in form)
    lead_in = int(0.25 * SR)  # small silence before first downbeat
    n = int(total_bars * bar_n) + lead_in + SR
    out = np.zeros(n)

    t_k = np.arange(int(0.35 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-t_k * 30)) / SR) * np.exp(-t_k * 9)
    hat = signal.lfilter(*signal.butter(2, 7000, "high", fs=SR), rng.standard_normal(int(0.04 * SR))) * _env(int(0.04 * SR), 0.001, 0.012)
    clap = signal.lfilter(*signal.butter(2, [900, 2500], "band", fs=SR), rng.standard_normal(int(0.15 * SR))) * _env(int(0.15 * SR), 0.002, 0.05)

    prog = [0, 8, 3, 10] if mode == "minor" else [0, 5, 9, 7]
    scale = [0, 2, 3, 5, 7, 8, 10] if mode == "minor" else [0, 2, 4, 5, 7, 9, 11]
    melody = rng.choice(scale, size=16)
    lp_bass = signal.butter(2, 400, "low", fs=SR)
    lp_pad = signal.butter(2, 1800, "low", fs=SR)
    formants = [signal.butter(2, [f * 0.85, f * 1.15], "band", fs=SR) for f in (800, 1150, 2900)]

    bar0 = 0
    for name, bars in form:
        layers = _LAYERS[name]
        for b in range(bar0, bar0 + bars):
            pos = lead_in + int(b * bar_n)
            prog_i = (b // 2) % 4
            chord_root = 48 + root_pc + prog[prog_i]
            # chord quality from the scale degree
            q = {"minor": {0: 3, 8: 4, 3: 4, 10: 4}, "major": {0: 4, 5: 4, 9: 3, 7: 4}}[mode][prog[prog_i]]
            progress = (b - bar0) / bars
            for k in range(4):
                bp = pos + int(k * beat * SR)
                if "kick" in layers:
                    _add(out, kick, bp, 0.9)
                if "clap" in layers and k in (1, 3):
                    _add(out, clap, bp, 0.35)
                if "hat" in layers:
                    _add(out, hat, bp + int(0.5 * beat * SR), 0.25)
            if "bass" in layers:
                ln = int(0.45 * beat * SR)
                note = _saw(_midi_hz(chord_root - 12), ln) * _env(ln, 0.005, 0.12)
                note = signal.lfilter(*lp_bass, note)
                for k in range(4):
                    _add(out, note, pos + int((k + 0.5) * beat * SR), 0.5)
            if "pad" in layers:
                ln = int(bar_n)
                pad = sum(_saw(_midi_hz(chord_root + iv), ln, d) for iv in (0, q, 7) for d in (-0.004, 0.004))
                pad = signal.lfilter(*lp_pad, pad) * np.minimum(np.arange(ln) / (0.1 * SR), 1)
                _add(out, pad, pos, 0.05)
            if "vocal" in layers:
                for k in range(2):
                    ln = int(2 * beat * SR * 0.9)
                    m = 60 + root_pc + melody[(b * 2 + k) % 16]
                    t = np.arange(ln) / SR
                    f = _midi_hz(m) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))
                    src = signal.square(2 * np.pi * np.cumsum(f) / SR, duty=0.3)
                    v = sum(signal.lfilter(*fm, src) * g for fm, g in zip(formants, (1.0, 0.7, 0.3)))
                    _add(out, v * _env(ln, 0.04, 0.8), pos + int(k * 2 * beat * SR), 0.18)
            if "roll" in layers:
                div = 2 if progress < 0.5 else (4 if progress < 0.85 else 8)
                for k in range(4 * div):
                    _add(out, clap, pos + int(k * beat / div * SR), 0.12 + 0.25 * progress)
            if "riser" in layers:
                ln = int(bar_n)
                noise = signal.lfilter(*signal.butter(2, 1500 + 6000 * progress, "high", fs=SR), rng.standard_normal(ln))
                _add(out, noise, pos, 0.03 + 0.12 * progress)
        bar0 += bars

    out = signal.lfilter(*signal.butter(1, 20, "high", fs=SR), out)
    out /= np.abs(out).max() + 1e-9
    stereo = np.stack([out, np.roll(out, 7)]) * 0.89
    return stereo.astype(np.float32)
