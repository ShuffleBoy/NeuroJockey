"""Set programs: pick tracks from a larger crate and order them along a section plan."""

import re
from dataclasses import dataclass

import numpy as np

from ..analysis import TrackAnalysis
from .compat import MAX_BEATMATCH, best_ratio, harmonic, usable_shift


@dataclass
class Section:
    name: str
    share: float  # fraction of the set's duration
    energy: tuple[float, float]  # target energy percentile at the start / end of the section
    kind: str  # "dance" | "rest"


PROGRAMS = {
    "club": [
        Section("Warm-up", 0.15, (0.40, 0.62), "dance"),
        Section("Dance I", 0.25, (0.68, 0.85), "dance"),
        Section("Breather", 0.13, (0.30, 0.22), "rest"),
        Section("Peak", 0.32, (0.78, 1.00), "dance"),
        Section("Closing", 0.15, (0.55, 0.30), "rest"),
    ],
    "warmup": [
        Section("Opening", 0.4, (0.2, 0.45), "rest"),
        Section("Groove", 0.6, (0.45, 0.7), "dance"),
    ],
    "peak": [
        Section("Peak", 0.75, (0.75, 1.0), "dance"),
        Section("Release", 0.25, (0.6, 0.4), "rest"),
    ],
}


def artists(t: TrackAnalysis) -> set[str]:
    head = t.title.split(" - ")[0] if " - " in t.title else ""
    return {a.strip().lower() for a in re.split(r",|&| feat\.? | ft\.? ", head) if a.strip()}


def song(t: TrackAnalysis) -> str:
    title = t.title.split(" - ", 1)[-1]
    return re.sub(r"\(.*?\)|\[.*?\]", "", title).strip().lower()


def play_minutes(t: TrackAnalysis, mode: str, hl_bars: int) -> float:
    """Rough on-air time of one track (overlaps removed)."""
    if mode == "full":
        bars = t.n_bars - 8
    else:
        bars = min(t.n_bars, hl_bars + 16) - 8
    return max(1.0, bars * t.bar_len / 60)


def energy_rank(tracks: list[TrackAnalysis]) -> dict[str, float]:
    lv = np.array([t.energy_level for t in tracks])
    ranks = lv.argsort().argsort() / max(1, len(tracks) - 1)
    return {t.track_id: float(r) for t, r in zip(tracks, ranks)}


def select_and_order(crate: list[TrackAnalysis], program: str, minutes: float, mode: str, hl_bars: int,
                     max_shift: int = 1, beam: int = 96, log=lambda *a: None) -> list[tuple[TrackAnalysis, str]]:
    """Choose tracks for each slot of the program; returns [(track, section name)] in play order."""
    sections = PROGRAMS[program]
    pool = [t for t in crate if not t.variable_tempo and t.n_bars >= 24]
    rank = energy_rank(pool)
    avg = float(np.median([play_minutes(t, mode, hl_bars) for t in pool]))
    k = max(2, min(len(pool), int(round(minutes / avg))))

    # slot targets: section of each slot and target energy (linear inside the section)
    slots = []
    for sec in sections:
        n = max(1, int(round(k * sec.share)))
        for j in range(n):
            f = j / max(1, n - 1)
            slots.append((sec, sec.energy[0] + (sec.energy[1] - sec.energy[0]) * f, j == 0))
    k = len(slots)
    log(f"programme '{program}': {k} slots from a crate of {len(pool)} (~{avg:.1f} min each)")

    n = len(pool)
    gap = np.zeros((n, n))
    key = np.zeros((n, n))
    for i, a in enumerate(pool):
        for j, b in enumerate(pool):
            if i != j:
                gap[i, j] = best_ratio(a.bpm, b.bpm)[1]
                key[i, j] = harmonic(a, b, usable_shift(b, max_shift))[1]
    art = [artists(t) for t in pool]
    songs = [song(t) for t in pool]

    def step_cost(prev: int | None, j: int, slot) -> float:
        sec, target, boundary = slot
        e = rank[pool[j].track_id]
        c = 2.0 * abs(e - target)
        mood = pool[j].mood or {}
        val, club = mood.get("valence", 0.5), mood.get("club", 0.5)
        if sec.kind == "dance":
            c += 2.5 * max(0.0, target - e)  # a dance slot must not sag below its energy
            # ... nor turn into a sad song: loud is not the same as danceable (aggressive phonk is fine,
            # a melancholic ballad is not, so club-fitness weighs more than happiness)
            fit = 0.6 * club + 0.4 * val
            c += 5.0 * max(0.0, 0.45 + 0.2 * target - fit)
        else:
            c += 0.6 * max(0.0, e - 0.5)  # rest slots: keep it mellow
            c += 0.8 * max(0.0, club - 0.8)
        if prev is None:
            return c
        g = gap[prev, j]
        blendable = g <= MAX_BEATMATCH
        # inside a section the mix should flow (blendable, close tempo); at a boundary a reset is fine
        c += (0.3 if boundary else 1.0) * (min(g, 0.4) * 3 + (0 if blendable else 0.6))
        c += (0.2 if boundary else 0.6) * (1 - key[prev, j])
        if art[prev] & art[j]:
            c += 2.0
        return c

    states = [(step_cost(None, j, slots[0]), [j]) for j in range(n)]
    states.sort(key=lambda s: s[0])
    states = states[:beam]
    for p in range(1, k):
        nxt = []
        for cost, path in states:
            used = set(path)
            used_songs = {songs[x] for x in path}
            counts: dict[str, int] = {}
            for x in path:
                for a in art[x]:
                    counts[a] = counts.get(a, 0) + 1
            for j in range(n):
                if j in used or songs[j] in used_songs or any(counts.get(a, 0) >= 2 for a in art[j]):
                    continue
                nxt.append((cost + step_cost(path[-1], j, slots[p]), path + [j]))
        if not nxt:
            break
        nxt.sort(key=lambda s: s[0])
        states = nxt[:beam]
    best = states[0][1]
    return [(pool[j], slots[p][0].name) for p, j in enumerate(best)]


def section_kinds(program: str) -> dict[str, str]:
    return {s.name: s.kind for s in PROGRAMS[program]}
