import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Segment:
    start_bar: int
    end_bar: int
    label: str  # intro | build | drop | break | groove | outro
    energy: float

    @property
    def bars(self) -> int:
        return self.end_bar - self.start_bar


@dataclass
class Highlight:
    start_bar: int
    end_bar: int
    score: float
    kind: str  # drop | chorus | peak


@dataclass
class TrackAnalysis:
    path: str
    track_id: str
    title: str
    duration: float
    bpm: float
    t0: float  # time (s) of bar 0 = first downbeat on the fitted grid
    n_bars: int
    variable_tempo: bool
    grid_error_ms: float
    key: str  # e.g. "A minor"
    key_pc: int  # tonic pitch class, 0 = C
    key_mode: str  # "major" | "minor"
    camelot: str  # e.g. "8A"
    key_confidence: float
    lufs: float
    energy_level: float  # 0..1 comparable across tracks
    bar_energy: list[float]
    bar_vocal: list[float] | None
    segments: list[Segment]
    highlights: list[Highlight]
    has_stems: bool = False
    bar_stem_db: dict[str, list[float]] | None = None  # per-bar RMS of each stem (dB)
    beat_vocal: list[float] | None = None  # per-beat vocal activity 0..1 (relative to the track's vocal level)
    vocal8: list[float] | None = None  # vocal stem level per 8th note, dB relative to its loud parts (breaths)
    chroma_bars: list[list[float]] | None = None  # per-bar pitch-class profile of the harmonic stems
    struct_version: int = 0
    mood: dict | None = None  # {"valence": 0..1, "dance": 0..1, "tags": [...], "src": "llm"|"audio"}
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    version: int = 0

    @property
    def beat_len(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_len(self) -> float:
        return 4 * 60.0 / self.bpm

    def segment_at(self, bar: int) -> Segment | None:
        for s in self.segments:
            if s.start_bar <= bar < s.end_bar:
                return s
        return None

    @property
    def first_drop(self) -> int | None:
        return next((s.start_bar for s in self.segments if s.label == "drop"), None)

    @property
    def outro_start(self) -> int | None:
        last = self.segments[-1] if self.segments else None
        return last.start_bar if last and last.label == "outro" else None

    def vocal_amount(self, start_bar: int, end_bar: int) -> float:
        if not self.bar_vocal:
            return 0.0
        part = self.bar_vocal[max(0, start_bar) : max(start_bar + 1, end_bar)]
        return float(sum(part) / len(part)) if part else 0.0

    def chroma(self, start_bar: int, end_bar: int):
        """Mean pitch-class profile over a bar range (None without chroma data)."""
        if not self.chroma_bars:
            return None
        part = self.chroma_bars[max(0, start_bar):max(start_bar + 1, end_bar)] or self.chroma_bars[-1:]
        return [sum(c[k] for c in part) / len(part) for k in range(12)]

    def vocal_db8(self, eighth: int) -> float:
        """Vocal level (dB re. its loud parts) on an 8th-note slot; 0 when unknown."""
        if self.vocal8 and 0 <= eighth < len(self.vocal8):
            return self.vocal8[eighth]
        return -60.0 if self.vocal8 else 0.0

    def breath_before(self, beat: int, bars: int = 4) -> float | None:
        """Track beats back from `beat` to the last breath (an 8th note clearly below the line around it)
        within `bars` bars; None if the singer never pauses there (or no 8th-note data)."""
        if not self.vocal8:
            return None
        e = 2 * beat
        lv = [self.vocal_db8(e - k) for k in range(1, 8 * bars + 1)]
        head = sorted(lv[:16])
        floor = max(-15.0, head[len(head) // 2] - 6.0)
        for k, v in enumerate(lv, 1):
            if v < floor:
                return k / 2
        return None

    def vocal_on(self, beat: int) -> float:
        """Vocal activity on a single beat (falls back to the bar value)."""
        if self.beat_vocal and 0 <= beat < len(self.beat_vocal):
            return self.beat_vocal[beat]
        if self.bar_vocal and 0 <= beat // 4 < len(self.bar_vocal):
            return self.bar_vocal[beat // 4]
        return 0.0

    def _drums_active(self) -> list[bool] | None:
        d = (self.bar_stem_db or {}).get("drums")
        if not d:
            return None
        ref = sorted(d)[int(len(d) * 0.7)]  # level of the "full" drum sections
        if ref < -35:
            return None  # no real drum track
        return [x > ref - 12 for x in d]

    @property
    def active_end(self) -> int:
        """Bar after the last one with the drums still playing (tracks often end on a drum-less tag)."""
        act = self._drums_active()
        if not act or not any(act):
            return self.n_bars
        return max(i for i, a in enumerate(act) if a) + 1

    @property
    def active_start(self) -> int:
        """First bar that is not near-silence."""
        d = self.bar_stem_db
        if not d:
            return 0
        loud = [max(v[i] for v in d.values()) for i in range(self.n_bars)]
        ref = sorted(loud)[len(loud) // 2]
        return next((i for i, x in enumerate(loud) if x > ref - 20), 0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackAnalysis":
        d = dict(d)
        d["segments"] = [Segment(**s) for s in d["segments"]]
        d["highlights"] = [Highlight(**h) for h in d["highlights"]]
        return cls(**d)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "TrackAnalysis":
        return cls.from_dict(json.loads(path.read_text()))
