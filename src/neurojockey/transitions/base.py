"""Transition plugin interface."""

from dataclasses import dataclass, field

import numpy as np

from ..analysis import TrackAnalysis
from ..engine.plan import MixPlan, SetTransition

REGISTRY: dict[str, type["Transition"]] = {}


@dataclass
class Ctx:
    a: TrackAnalysis
    b: TrackAnalysis
    key: float  # harmonic compatibility after pitch shifts, 0..1
    bpm_gap: float  # relative tempo gap after half/double-time choice
    energy_delta: float  # b.energy_level - a.energy_level
    stems: bool  # both tracks have stems
    a_vocal_tail: float
    b_vocal_head: float
    b_clean_intro: bool
    mode: str  # "full" | "highlights"
    recent: list[str] = field(default_factory=list)
    rest: bool = False  # the incoming track belongs to a rest / breather section
    room: float = 1e9  # longest overlap (output bars) the two regions allow
    groove_change: bool = False  # one deck plays half/double time against the other

    @property
    def can_beatmatch(self) -> bool:
        return self.bpm_gap <= 0.17 and not (self.a.variable_tempo or self.b.variable_tempo)

    @property
    def stretch_penalty(self) -> float:
        """1.0 for matched tempos, falling as both tracks have to be stretched more to meet."""
        return float(max(0.5, 1 - max(0.0, self.bpm_gap - 0.06) * 2.5))


@dataclass
class Window:
    plan: MixPlan
    ai: int  # outgoing slot index
    bi: int  # incoming slot index
    w: float  # global beat where B starts
    length: float  # overlap in beats
    tr: SetTransition

    @property
    def end(self) -> float:
        return self.w + self.length

    def at(self, bars: float) -> float:
        """Global beat `bars` bars into the window (negative = before W), never past the window end:
        a transition written for 16 bars still has to make sense when the planner squeezes it."""
        return min(self.w + 4 * bars, self.end)

    def frac(self, f: float) -> float:
        """Global beat at fraction `f` of the overlap (0 = W, 1 = window end)."""
        return self.w + self.length * f

    def a(self, lane, points):
        self.plan.automate(self.ai, lane, points)

    def b(self, lane, points):
        self.plan.automate(self.bi, lane, points)


@dataclass
class Handoff:
    """Where the builder hands the lead vocal over (global beats). A's singer stops at a line end in
    [lo, hi], as close to `target` as the phrasing allows; B's singer starts at its next line, not before
    `b_min`. b="keep": B's vocal lane stays as the transition wrote it (e.g. B sings only on its drop)."""
    target: float
    lo: float
    hi: float
    b_min: float | None = None
    b: str = "line"


class Transition:
    name = "base"
    title = ""
    description = ""
    beatmatched = True  # needs tempo-matched overlap
    needs_stems = False
    pre_bars = 0  # bars of A's tail used before W
    harmonic_overlap = True  # do both tracks' tonal parts sound together? (drives key checks / shifts)
    tail_beats = 0  # A plays this many beats past its phrase point (e.g. echo out "on the one")
    tempo_bridge_beats = 0  # non-beatmatched: ramp the master tempo to B's over A's last N beats
    min_bars = 4  # shortest overlap that still sounds like this technique
    fx_tail = False  # an echo / reverb tail carries A's last word out (softens a mid-line exit)

    def bars(self, ctx: Ctx) -> int:
        return 16

    def entry(self, ctx: Ctx) -> str:
        """Where B enters: "intro" (before its highlight/at its start) or "drop" (straight into the drop)."""
        return "intro"

    def score(self, ctx: Ctx) -> float:
        return 0.0

    def handoff(self, w: float, length: float, params: dict) -> Handoff | None:
        """Vocal handoff window for the builder; None = the transition leaves vocals alone."""
        return None

    @staticmethod
    def mid_handoff(w: float, length: float) -> Handoff:
        return Handoff(w + length / 2, w + length / 4, w + length * 3 / 4)

    def apply(self, win: Window) -> None:
        raise NotImplementedError

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.name != "base":
            REGISTRY[cls.name] = cls


EPS = 0.01


def ramp(b0: float, b1: float, v0: float, v1: float, curve: str = "lin", n: int = 24) -> list[tuple[float, float]]:
    """Automation points from (b0, v0) to (b1, v1).
    curve: lin | in (equal-power rise, use for fade-ins) | out (equal-power fall) | exp | log"""
    if b1 < b0:
        raise ValueError(f"ramp runs backwards ({b0} -> {b1})")
    x = np.linspace(0, 1, n)
    if curve == "in":
        vals = v0 + (v1 - v0) * np.sin(x * np.pi / 2)
    elif curve == "out":
        vals = v1 + (v0 - v1) * np.cos(x * np.pi / 2)
    else:
        s = {"lin": x, "exp": x**2, "log": np.sqrt(x)}[curve]
        vals = v0 + (v1 - v0) * s
    return [(float(b0 + (b1 - b0) * xi), float(v)) for xi, v in zip(x, vals)]


def step(beat: float, before: float, after: float) -> list[tuple[float, float]]:
    return [(beat - EPS, before), (beat, after)]

