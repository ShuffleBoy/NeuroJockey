"""Pairwise compatibility: tempo ratio, harmonic shift, energy."""

from ..analysis.key import chroma_fit, key_compat, shift_camelot

MAX_STRETCH = 0.08  # comfortable stretch for one track
MAX_BEATMATCH = 0.17  # tempo gap that can still be beatmatched when both tracks meet in the middle


def best_ratio(prev_bpm: float, bpm: float) -> tuple[float, float]:
    """Pick 1x / half / double time for `bpm` against `prev_bpm`. Returns (ratio, relative gap)."""
    best = (1.0, abs(bpm / prev_bpm - 1))
    for r in (0.5, 2.0):
        eff = bpm * r
        if 60 <= eff <= 200:
            gap = abs(eff / prev_bpm - 1)
            if gap <= MAX_BEATMATCH and gap + 0.03 < best[1]:
                best = (r, gap)
    return best


def best_shift(cam_a: str, cam_b: str, max_shift: int = 1) -> tuple[int, float]:
    """Semitone shift for B that best fits A harmonically. Returns (shift, compat after shift)."""
    base = key_compat(cam_a, cam_b)
    best = (0, base)
    for s in (1, -1, 2, -2):
        if abs(s) > max_shift:
            continue
        c = key_compat(cam_a, shift_camelot(cam_b, s))
        if c - 0.1 * abs(s) > best[1] - 0.1 * abs(best[0]) and c >= base + 0.25:
            best = (s, c)
    return best


def usable_shift(b, max_shift: int) -> int:
    """A sung track is never pitch-shifted (see the planner), so ordering must not count on it."""
    return 0 if getattr(b, "bar_vocal", None) and b.vocal_amount(0, b.n_bars) > 0.3 else max_shift


def pair_cost(a, b, max_shift: int) -> float:
    _, gap = best_ratio(a.bpm, b.bpm)
    _, key = harmonic(a, b, usable_shift(b, max_shift))
    tempo = min(gap / MAX_STRETCH, 3.0)
    return 1.2 * tempo + 1.0 * (1 - key)


def harmonic(a, b, max_shift: int = 1, a_bars: tuple[int, int] | None = None,
             b_bars: tuple[int, int] | None = None) -> tuple[int, float]:
    """Semitone shift for B and the resulting harmonic compatibility (0..1).

    Blends the Camelot rule with the measured notes of the two parts that will actually overlap
    (A's exit, B's entry); falls back to Camelot alone without chroma data."""
    pa = a.chroma(*(a_bars or (0, a.n_bars))) if hasattr(a, "chroma") else None
    pb = b.chroma(*(b_bars or (0, b.n_bars))) if hasattr(b, "chroma") else None

    def fit(s):
        cam = key_compat(a.camelot, shift_camelot(b.camelot, s))
        if pa is None or pb is None:
            return cam
        return 0.4 * cam + 0.6 * chroma_fit(pa, pb, s)

    base = fit(0)
    best = (0, base)
    for s in (1, -1, 2, -2):
        if abs(s) > max_shift:
            continue
        c = fit(s)
        if c - 0.08 * abs(s) > best[1] - 0.08 * abs(best[0]) and c >= base + 0.15:
            best = (s, c)
    return best
