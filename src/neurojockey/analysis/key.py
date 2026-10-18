"""Key detection (Krumhansl/Temperley profiles on CQT chroma) and Camelot wheel helpers."""

import numpy as np

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

_PROFILES = {
    "major": [
        np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
        np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
    ],
    "minor": [
        np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
        np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
    ],
}


def key_from_profile(profile: np.ndarray) -> tuple[int, str, float]:
    """Key of a 12-bin pitch-class profile: (tonic pitch class, mode, confidence 0..1)."""
    profile = np.asarray(profile, float)
    profile = profile / (profile.sum() + 1e-9)
    scores = []
    for mode, profs in _PROFILES.items():
        for pc in range(12):
            r = np.mean([np.corrcoef(profile, np.roll(p, pc))[0, 1] for p in profs])
            scores.append((r, pc, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    confidence = float(np.clip((best[0] - second[0]) * 5 + best[0] * 0.5, 0, 1))
    return best[1], best[2], confidence


def key_name(pc: int, mode: str) -> str:
    return f"{NOTE_NAMES[pc % 12]} {mode}"


def camelot(pc: int, mode: str) -> str:
    major_pc = pc % 12 if mode == "major" else (pc + 3) % 12
    num = (major_pc * 7 % 12 + 7) % 12 + 1
    return f"{num}{'B' if mode == 'major' else 'A'}"


def parse_camelot(code: str) -> tuple[int, str]:
    return int(code[:-1]), code[-1]


def shift_camelot(code: str, semitones: int) -> str:
    num, letter = parse_camelot(code)
    return f"{(num - 1 + 7 * semitones) % 12 + 1}{letter}"


def key_compat(a: str, b: str) -> float:
    """Harmonic compatibility of two Camelot codes, 1 = perfect."""
    na, la = parse_camelot(a)
    nb, lb = parse_camelot(b)
    d = (nb - na) % 12
    if la == lb:
        return {0: 1.0, 1: 0.9, 11: 0.9, 2: 0.55, 10: 0.5, 7: 0.45, 5: 0.3}.get(d, 0.1)
    if d == 0:
        return 0.85
    # diagonal moves (e.g. 8A -> 9B, 8B -> 7A)
    if (la == "A" and d == 1) or (la == "B" and d == 11):
        return 0.6
    return 0.15


def chroma_fit(pa, pb, shift: int = 0) -> float:
    """How well two pitch-class profiles sound together (B transposed by `shift` semitones), 0..1.
    Measured on the actual notes rather than the Camelot labels, so a relative-major/minor mix-up
    in key detection does not matter."""
    a = np.asarray(pa, float)
    b = np.roll(np.asarray(pb, float), shift)
    a, b = a - a.mean(), b - b.mean()
    r = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return float(np.clip((r - 0.1) / 0.75, 0, 1))
