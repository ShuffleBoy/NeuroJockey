"""Data model: SetList (editable set) and MixPlan (slots, tempo, automation; positions in global beats)."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Lane names. Values are linear multipliers unless noted.
GAIN = "gain"  # channel fader
EQ_LOW, EQ_MID, EQ_HIGH = "eq_low", "eq_mid", "eq_high"
FILTER = "filter"  # -1 = LPF fully closed .. 0 = off .. +1 = HPF fully closed
SEND_REVERB = "send_reverb"
# vocal-only buses for a line cut without a pause (raw vocal stem, before its own stem mute):
SEND_VREVERB, SEND_VECHO = "send_vreverb", "send_vecho"  # into a reverb / a tempo echo
VOCAL_DRY, VOCAL_FILTER = "vocal_dry", "vocal_filter"  # a dry copy through its own filter (knob as FILTER)
SEND_VSLOW = "send_vslow"  # the voice slows down like a stopping tape (pitch falls) into a reverb
SEND_VSWELL = "send_vswell"  # B's first word reverbed and reversed: a swell that ends where the word starts
VOCAL_BUSES = (SEND_VREVERB, SEND_VECHO, VOCAL_DRY, SEND_VSLOW, SEND_VSWELL)
PITCH = "pitch"  # semitones; overrides Slot.pitch_shift when present
STEM = "stem:"  # prefix, e.g. "stem:vocals"
NEUTRAL = {GAIN: 1.0, EQ_LOW: 1.0, EQ_MID: 1.0, EQ_HIGH: 1.0, FILTER: 0.0, SEND_REVERB: 0.0, SEND_VREVERB: 0.0, SEND_VECHO: 0.0, VOCAL_DRY: 0.0, VOCAL_FILTER: 0.0, SEND_VSLOW: 0.0, SEND_VSWELL: 0.0, PITCH: 0.0}


def neutral(lane: str) -> float:
    return 1.0 if lane.startswith(STEM) else NEUTRAL[lane]


@dataclass
class SetItem:
    path: str
    track_id: str
    title: str
    start_bar: int  # in the track's own bars
    end_bar: float
    ratio: float = 1.0  # tempo multiplier for half/double-time mixing (0.5, 1, 2)
    pitch_shift: int = 0  # semitones (harmonic mixing)
    section: str = ""  # programme section, e.g. "Peak"


@dataclass
class SetTransition:
    type: str
    bars: float  # overlap in output bars (0 = butt splice, negative = gap)
    params: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class SetList:
    items: list[SetItem]
    transitions: list[SetTransition]
    notes: str = ""

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "SetList":
        d = json.loads(Path(path).read_text())
        return cls(
            items=[SetItem(**x) for x in d["items"]],
            transitions=[SetTransition(**x) for x in d["transitions"]],
            notes=d.get("notes", ""),
        )


@dataclass
class Segment:
    """Maps output beats [out, out+length) of a slot onto source beats starting at src."""

    out: float  # relative to slot start, in beats
    length: float
    src: float  # source beat on the track's (ratio-adjusted) grid


@dataclass
class Slot:
    path: str
    track_id: str
    title: str
    bpm: float  # effective bpm = track bpm * ratio
    t0: float  # time of source beat 0
    start: float  # global beat where the slot begins
    beats: float  # slot length in output beats
    src_beat: float  # first source beat (effective grid)
    pitch_shift: int = 0
    gain_db: float = 0.0  # loudness normalisation
    segments: list[Segment] = field(default_factory=list)

    @property
    def end(self) -> float:
        return self.start + self.beats

    def default_segments(self) -> list[Segment]:
        return self.segments or [Segment(0.0, self.beats, self.src_beat)]


@dataclass
class MixPlan:
    slots: list[Slot]
    tempo: list[tuple[float, float]]  # (global beat, bpm), piecewise linear
    lanes: dict[str, list[tuple[float, float]]] = field(default_factory=dict)  # "slot:lane" -> points
    master_fx: list[dict] = field(default_factory=list)
    fx: dict[str, dict] = field(default_factory=dict)  # per-slot fx settings, key = slot index
    events: list[dict] = field(default_factory=list)  # {"beat", "text"} for the tracklist

    def automate(self, slot: int, lane: str, points: list[tuple[float, float]]) -> None:
        if lane not in NEUTRAL and not lane.startswith(STEM):
            raise ValueError(f"unknown lane {lane}")
        key = f"{slot}:{lane}"
        self.lanes.setdefault(key, []).extend((float(b), float(v)) for b, v in points)
        self.lanes[key].sort(key=lambda p: p[0])

    def clear(self, slot: int, lane: str, b0: float, b1: float) -> None:
        """Drop a lane's points in [b0, b1] (to rewrite that stretch)."""
        key = f"{slot}:{lane}"
        if key in self.lanes:
            self.lanes[key] = [p for p in self.lanes[key] if not b0 <= p[0] <= b1]

    def lanes_for(self, slot: int) -> dict[str, list[tuple[float, float]]]:
        prefix = f"{slot}:"
        return {k[len(prefix):]: v for k, v in self.lanes.items() if k.startswith(prefix)}

    @property
    def total_beats(self) -> float:
        return max(s.end for s in self.slots)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))
