"""The transition library. Each class = one DJ technique."""

from ..engine.plan import (EQ_HIGH, EQ_LOW, FILTER, GAIN, SEND_REVERB, Segment)
from .base import Ctx, Handoff, Transition, Window, ramp, step

V, D, B_, O = "stem:vocals", "stem:drums", "stem:bass", "stem:other"


FX_MOVES = {"gate", "loop_roll", "riser", "cut", "reverb_wash"}


def _b_lead_drums(ctx: Ctx) -> float:
    """Share of the 8 bars before B's drop / highlight where B's drums play. Drums-first moves (drum bed, stem
    relay) hand the groove to B's drums: over a drum-less intro they leave a hole."""
    b = ctx.b
    act = b._drums_active()
    if act is None:
        return 1.0  # unknown: do not block
    target = b.first_drop if ctx.mode == "full" else (b.highlights[0].start_bar if b.highlights else None)
    if not target:
        return 1.0
    lead, last = act[max(0, target - 8):target], act[max(0, target - 4):target]
    if not lead:
        return 1.0
    # a pre-drop drum break (drums out for the last bars) leaves the bed empty right before the drop
    return min(sum(lead) / len(lead), sum(last) / len(last))


def _drum_swap(win: Window, at: float) -> None:
    """Two rap drum patterns (hats, snares, swing) layered sound like drums in disarray even with the kicks
    locked: B's drums wait, and the kits swap on the downbeat together with the bass (stems only)."""
    win.b(D, [(win.w, 0.0)] + step(at, 0, 1))
    win.a(D, step(at, 1, 0))


def _fx_spacing(ctx: Ctx, name: str) -> float:
    """Effect moves are seasoning: at most about one in six transitions."""
    return 0.3 if name in FX_MOVES and any(n in FX_MOVES for n in ctx.recent[-6:]) else 1.0


def _freshness(ctx: Ctx, name: str) -> float:
    return (0.25 if name in ctx.recent[-2:] else 1.0) * _fx_spacing(ctx, name)


def ctx_stems(win: Window) -> bool:
    from .. import stems as stemlib

    return stemlib.has_stems(win.plan.slots[win.ai].track_id)


def _land(win: Window, start_filter: float, bars: float = 2) -> None:
    """Soft landing for B entering on a drop: start from (roughly) the filter A ended on and open it
    up, with a short gain rise, so B blooms in instead of slamming. `landing_bars` in the set file
    overrides the length (0 = hard slam)."""
    bars = win.tr.params.get("landing_bars", bars)
    if bars <= 0:
        return
    end = win.w + 4 * bars
    win.b(FILTER, ramp(win.w, end, start_filter, 0, "log"))
    win.b(GAIN, ramp(win.w, win.w + min(4, 4 * bars), 0.6, 1, "in"))


# --- classic beatmatched -------------------------------------------------------------------------


class Blend(Transition):
    name = "blend"
    title = "Beatmatch blend"
    description = "Long equal-power crossfade on matched tempo with a smooth bass handoff in the middle."

    def bars(self, ctx):
        return 32 if ctx.mode == "full" and ctx.b_clean_intro else 16

    def score(self, ctx):
        if not ctx.can_beatmatch or ctx.key < 0.5:
            return 0  # an equal-power fade layers both tracks' music: keys must agree
        # by ear ("каша"): a full-track crossfade only works between quiet, instrumental parts - B's clean
        # intro under A without vocals on either side; otherwise the bass swap / drum bed do it cleanly
        if not ctx.b_clean_intro or ctx.a_vocal_tail > 0.3 or ctx.b_vocal_head > 0.3 or ctx.bpm_gap > 0.04:
            return 0
        s = 1.0 * (0.2 + 0.8 * ctx.key)
        return (s * (1.2 if ctx.b_clean_intro else 1.0) * (1.5 if ctx.rest else 1.0)
                * _freshness(ctx, self.name) * ctx.stretch_penalty)

    def handoff(self, w, length, params):
        return self.mid_handoff(w, length)

    def apply(self, win: Window):
        mid = win.w + win.length / 2
        win.a(GAIN, ramp(win.w, win.end, 1, 0, "out"))
        win.b(GAIN, ramp(win.w, win.end, 0, 1, "in"))
        win.a(EQ_LOW, ramp(mid - 2, mid + 2, 1, 0))
        win.b(EQ_LOW, [(win.w, 0.0)] + ramp(mid - 2, mid + 2, 0, 1))


class BassSwap(Transition):
    name = "bass_swap"
    title = "Bass swap"
    description = "B fades in with its lows killed; on the phrase downbeat the basslines swap instantly."

    def bars(self, ctx):
        return 16 if ctx.key >= 0.6 else 8  # clashing music: layer it briefly

    def score(self, ctx):
        if not ctx.can_beatmatch:
            return 0
        clash = 0.5 if ctx.key < 0.3 else 1.0  # two keys over a beat: a clean cut is the better call
        return 1.15 * clash * (0.25 + 0.75 * ctx.key) * (1.2 if ctx.b_clean_intro else 1.0) * _freshness(ctx, self.name) * ctx.stretch_penalty

    def handoff(self, w, length, params):
        return self.mid_handoff(w, length)

    def apply(self, win: Window):
        mid = win.w + win.length / 2
        win.b(GAIN, ramp(win.w, mid, 0, 1, "in"))
        win.b(EQ_LOW, [(win.w, 0.0)] + step(mid, 0, 1))
        # keys that only half agree: A's bassline leaves early (a few bars with no bass beat a clash)
        a_low_out = win.frac(0.25) if win.tr.params.get("_key", 1.0) < 0.8 else mid
        win.a(EQ_LOW, step(a_low_out, 1, 0))
        win.a(EQ_HIGH, ramp(win.w, mid, 1, 0.5))
        win.a(GAIN, ramp(mid, win.end, 1, 0, "out"))
        _drum_swap(win, mid)


class FilterSweep(Transition):
    name = "filter"
    title = "Filter sweep"
    description = "Low-pass closes on A while a high-pass opens on B: A dissolves, B emerges."

    def bars(self, ctx):
        return 16 if ctx.key >= 0.6 else 8

    def score(self, ctx):
        if not ctx.can_beatmatch:
            return 0
        # both tracks' mids sound together half way through the sweep: needs agreeing keys
        if ctx.key < 0.5:
            return 0
        return 0.85 * (0.3 + 0.7 * ctx.key) * (1.25 if ctx.rest else 1.0) * _freshness(ctx, self.name) * ctx.stretch_penalty

    def handoff(self, w, length, params):
        return self.mid_handoff(w, length)

    def apply(self, win: Window):
        mid = win.w + win.length / 2
        win.a(FILTER, ramp(win.w, win.end, 0, -0.85, "exp"))
        win.b(FILTER, ramp(win.w, win.end, 0.85, 0, "log"))
        _drum_swap(win, mid)
        win.b(GAIN, ramp(win.w, win.at(4), 0, 1, "in"))
        win.a(GAIN, ramp(win.at(win.length / 4 * 0.75), win.end, 1, 0, "out"))
        win.a(EQ_LOW, step(mid, 1, 0))
        win.b(EQ_LOW, [(win.w, 0.0)] + step(mid, 0, 1))


class GateOut(Transition):
    name = "gate"
    title = "Trance gate out"
    description = "A gets chopped by a 1/8 gate with lows removed while B rises underneath."

    def bars(self, ctx):
        return 4

    def score(self, ctx):
        if not ctx.can_beatmatch:
            return 0
        # a trance-gate effect: chopping a rapper's line into 1/8s is a gimmick - rare, and not over vocals
        rare = (0.2 if self.name in ctx.recent[-8:] else 1.0) * _fx_spacing(ctx, self.name)
        return (0.25 * (1.3 if ctx.energy_delta > 0.05 else 1.0) * (1 - 0.8 * min(1.0, ctx.a_vocal_tail))
                * rare * ctx.stretch_penalty)

    def handoff(self, w, length, params):
        return self.mid_handoff(w, length)

    def apply(self, win: Window):
        pts = []
        n = int(win.length * 2)
        for k in range(n):
            b = win.w + k * 0.5
            depth = min(1.0, 0.3 + k / n)
            pts += [(b, 1.0), (b + 0.24, 1.0), (b + 0.25, 1 - depth), (b + 0.49, 1 - depth)]
        pts.append((win.end, 0.0))
        win.a(GAIN, pts)
        win.a(EQ_LOW, ramp(win.w, win.w + 4, 1, 0))
        win.b(GAIN, ramp(win.w, win.end, 0.2, 1, "in"))
        win.b(EQ_LOW, [(win.w, 0.0)] + step(win.w + 4, 0, 1))


# --- cuts and effects (tempo can jump) ------------------------------------------------------------


class LoopRoll(Transition):
    name = "loop_roll"
    title = "Loop roll"
    description = ("A's last bar loops and shrinks: 1 bar -> 1 beat -> 1/16 -> 1/32 with a rising high-pass, "
                   "the loop accelerates to B's tempo, then B drops.")
    beatmatched = False
    pre_bars = 2
    tempo_bridge_beats = 8

    def bars(self, ctx):
        return 0

    def entry(self, ctx):
        return "drop"

    def score(self, ctx):
        vocal_heavy = min(1.0, ctx.a_vocal_tail)
        if vocal_heavy > 0.3:
            return 0  # by ear: not for vocal tracks (an EDM roll under a rapper sounds amateur)
        return (0.45 * (1.4 if ctx.energy_delta > 0 else 0.8) * (1.3 if not ctx.can_beatmatch else 1)
                * (1 - 0.8 * vocal_heavy) * _freshness(ctx, self.name))

    def handoff(self, w, length, params):
        # by ear ("резко вокал обрывается"): A's vocal was muted mid-word where the roll begins; the
        # singer now finishes a line in the 2 bars before it. B sings when its drop hits
        return Handoff(w - 8.5, w - 16, w - 8.5, b="keep")

    def apply(self, win: Window):
        slot = win.plan.slots[win.ai]
        roll_out = slot.beats - 8
        src = slot.src_beat + roll_out
        segs = [Segment(0.0, roll_out, slot.src_beat)]
        pos = roll_out
        for length, count in ((4, 1), (1, 2), (0.25, 4), (0.125, 8)):
            for _ in range(count):
                segs.append(Segment(pos, length, src))
                pos += length
        slot.segments = segs
        win.a(FILTER, ramp(win.w - 8, win.w, 0, 0.55, "exp"))
        win.a(SEND_REVERB, ramp(win.w - 2, win.w, 0, 0.7))
        if ctx_stems(win):
            # roll the instrumental only - a stuttering syllable sounds amateur
            roll_start = win.w - 8
            win.a(V, ramp(roll_start - 0.5, roll_start + 0.5, 1, 0))
        win.plan.fx[str(win.ai)] = {"reverb_size": 0.75}
        win.plan.master_fx.append({"type": "impact", "at": win.w, "gain": 0.15})
        # by ear ("резко запускается", 8 of 10 rolls 'резко'): B blooms in over a phrase, not 2 bars
        _land(win, 0.55, 4)


class Cut(Transition):
    name = "cut"
    title = "Slam cut"
    description = "Hard phrase-aligned cut from the end of A's phrase straight into B's drop."
    beatmatched = False

    def bars(self, ctx):
        return 0

    def entry(self, ctx):
        return "drop"

    def score(self, ctx):
        # by ear every slam scored 1/5 ("резко", "вокал обрезан"): only when asked for explicitly
        return 0

    def apply(self, win: Window):
        win.a(EQ_LOW, ramp(win.w - 1, win.w, 1, 0.3))
        _land(win, 0.3, 1)


class ReverbWash(Transition):
    name = "reverb_wash"
    title = "Reverb wash"
    description = "A drowns in a long reverb up to its phrase end; B rises out of the tail with an opening high-pass."
    beatmatched = False
    pre_bars = 2
    harmonic_overlap = False  # only a diffuse tail overlaps B
    fx_tail = True

    def bars(self, ctx):
        return 0  # the reverb tail rings past A's end on its own (rendered beyond the slot)

    def score(self, ctx):
        # by ear it rarely fits these tracks: an occasional colour for a tempo jump, never where the
        # tracks could be blended (the phrase cut is the default for jumps)
        if ctx.can_beatmatch:
            return 0
        s = 0.5 * (1.15 if ctx.energy_delta < 0 else 1.0)
        return s * (0.2 if self.name in ctx.recent[-8:] else 1.0) * _fx_spacing(ctx, self.name)

    def handoff(self, w, length, params):
        # A's line ends before it drowns; by ear ("вход вокал стрем") B's voice creeping up through the
        # filtered rise is ugly: it waits for its first line once B is nearly open
        return Handoff(w - 1, w - 8, w - 0.5, b_min=w + 4)

    def apply(self, win: Window):
        cut = win.plan.slots[win.ai].end
        win.a(SEND_REVERB, ramp(cut - 8, cut, 0, 1, "exp"))
        win.a(GAIN, ramp(cut - 1, cut, 1, 0.2))
        win.b(GAIN, ramp(win.w, win.w + 8, 0.3, 1, "in"))
        win.b(FILTER, ramp(win.w, win.w + 8, 0.6, 0, "log"))
        win.plan.fx[str(win.ai)] = {"reverb_size": 0.95}


class RiserSlam(Transition):
    name = "riser"
    title = "Riser build"
    description = "8 bars of build on A: high-pass climbs, noise riser and snare roll, impact into B's drop."
    beatmatched = False
    pre_bars = 8

    def bars(self, ctx):
        return 0

    def entry(self, ctx):
        return "drop"

    def score(self, ctx):
        # EDM build (noise riser, snare roll, impact): seasoning on vocal pop / rap, not a staple.
        # By ear ("не к месту"): a build that lands on a drop no bigger than A's is a false promise
        if ctx.energy_delta <= 0.03:
            return 0
        return 0.12 * 1.6 * _freshness(ctx, self.name)

    def apply(self, win: Window):
        start = win.w - 32
        win.a(FILTER, ramp(start, win.w, 0, 0.7, "exp"))
        win.a(EQ_LOW, ramp(win.w - 8, win.w, 1, 0))
        win.plan.master_fx += [
            {"type": "riser", "start": start, "end": win.w, "gain": 0.22},
            {"type": "roll", "start": win.w - 16, "end": win.w, "gain": 0.25},
            {"type": "impact", "at": win.w, "gain": 0.3},
        ]
        _land(win, 0.2, 2)  # by ear 'резко': a 1-bar landing slammed


# --- stem-based -------------------------------------------------------------------------------------


class StemRelay(Transition):
    name = "stem_relay"
    title = "Stem relay"
    description = "Stems hand over one by one: B drums join, basslines swap, then music, then vocals."
    needs_stems = True
    min_bars = 8

    def bars(self, ctx):
        return 32 if ctx.mode == "full" else 16

    def score(self, ctx):
        if not (ctx.stems and ctx.can_beatmatch):
            return 0
        if ctx.key < 0.5 or _b_lead_drums(ctx) < 0.5:
            return 0  # the music stems overlap for half the relay; B's drums must carry the first half
        return 1.1 * (0.3 + 0.7 * ctx.key) * _freshness(ctx, self.name) * ctx.stretch_penalty

    def handoff(self, w, length, params):
        # by ear (4 of 9: "вокал долго вступал", "противно нарастает"): B's voice used to fade in over the
        # last eighth of the relay. Now the singers swap on a line right after B's music is in
        return Handoff(w + length * 5 / 8, w + length / 2, w + length * 13 / 16)

    def apply(self, win: Window):
        q = win.length / 4
        w = win.w
        win.b(D, ramp(w, w + q / 2, 0, 1, "in"))
        win.b(EQ_LOW, [(w, 0.25)] + step(w + q, 0.25, 1))  # no double kick before the swap
        win.b(B_, [(w, 0.0)] + step(w + q, 0, 1))
        win.a(B_, step(w + q, 1, 0))
        win.a(D, step(w + 2 * q, 1, 0))
        win.b(O, [(w, 0.0)] + ramp(w + 2 * q, w + 2.5 * q, 0, 1, "in"))
        win.a(O, ramp(w + 2 * q, w + 2.5 * q, 1, 0, "out"))
        # (without 8th-note vocal data; otherwise the builder's line-aware handoff replaces these)
        win.a(V, ramp(w + 2.5 * q - 1, w + 2.5 * q, 1, 0, "out"))
        win.b(V, [(w, 0.0)] + ramp(w + 2.5 * q, w + 2.5 * q + 1, 0, 1, "in"))
        win.a(GAIN, ramp(w + 3.5 * q, win.end, 1, 0, "out"))


class AcapellaBridge(Transition):
    name = "acapella"
    title = "Acapella bridge"
    description = "A's instrumental drops away and its vocal rides over B's instrumental before B's own vocal returns."
    needs_stems = True
    min_bars = 8

    def bars(self, ctx):
        return 8

    def score(self, ctx):
        if not (ctx.stems and ctx.can_beatmatch) or ctx.a_vocal_tail < 0.3 or ctx.key < 0.8:
            return 0
        # by ear, twice: A's vocal riding over B's music for long "слишком растянут" - a rare colour only
        return 0.3 * ctx.key * (0.6 + ctx.a_vocal_tail) * _freshness(ctx, self.name) * ctx.stretch_penalty

    def apply(self, win: Window):
        # positions are fractions of the overlap, so a squeezed window keeps the same shape
        w, f = win.w, win.frac
        win.b(GAIN, ramp(w, f(0.25), 0, 1, "in"))
        win.b(V, [(w, 0.0)] + ramp(f(0.75), win.end, 0, 1, "in"))
        win.b(EQ_LOW, [(w, 0.0)] + step(f(0.25), 0, 1))
        win.a(B_, step(f(0.25), 1, 0))
        win.a(D, ramp(w, f(0.25), 1, 0, "out"))
        win.a(O, ramp(f(0.125), f(0.5), 1, 0, "out"))
        win.a(V, ramp(f(0.75), win.end, 1, 0, "out"))


class DrumBed(Transition):
    name = "drum_bed"
    title = "Drum bed"
    description = ("B's drums slide in under A; A's music and vocal leave before B's bass, music and vocal "
                   "arrive, so the two keys never sound together.")
    needs_stems = True
    min_bars = 8
    harmonic_overlap = False

    def bars(self, ctx):
        return 16

    def score(self, ctx):
        if not (ctx.stems and ctx.can_beatmatch) or _b_lead_drums(ctx) < 0.5:
            return 0  # a drum bed needs B's drums in the bars it enters on
        if ctx.groove_change or ctx.bpm_gap > 0.02:
            return 0  # by ear: two kits for 16 bars over a half/double-time or stretched pair = "в разнобой"
        # percussive overlap: the less compatible the keys, the more this is the right call
        return 0.8 * (1.6 - 0.6 * ctx.key) * _freshness(ctx, self.name) * ctx.stretch_penalty

    def handoff(self, w, length, params):
        if params.get("_b_intro_vocal", 0.0) >= 0.5:
            # B's intro is sung: A's singer finishes the line around the window start (within a bar), B
            # starts on its next. By ear a clean stop at W is good ("вокал четко обрезан")
            return Handoff(w, w - 4, w + 4)
        # B sings only on its drop (window end): A's vocal ends on a line in the second half
        return Handoff(w + length * 3 / 4, w + length / 2, w + length - 2, b="keep")

    def apply(self, win: Window):
        # A's drums carry under B to the end ("круто, плавно, хочу чаще"): they cover B's own drum breaks.
        # Swapping the kits half-way left holes there and B's music crossing in over 2 bars left a thin
        # bar - both rated worse. Vocals are handed over on a line by the builder (V lanes = fallback).
        w, n = win.w, win.length / 4  # n bars
        q = lambda f: win.at(n * f)
        win.b(D, ramp(w, q(0.25), 0, 1, "in"))
        win.b(EQ_LOW, [(w, 0.25)] + step(q(0.5), 0.25, 1))  # no double kick before the swap
        win.b(B_, [(w, 0.0)] + step(q(0.5), 0, 1))
        win.a(B_, step(q(0.5), 1, 0))
        win.a(D, ramp(win.end - 2, win.end, 1, 0))
        if win.tr.params.get("_b_intro_vocal", 0.0) >= 0.5:
            # B's intro is sung: heard from the first beat; A's music leaves on the phrase (window start),
            # A's drums carry the groove under B's intro, basses swap half-way
            win.a(V, ramp(w - 0.5, w + 0.5, 1, 0))
            win.a(O, ramp(w, win.at(1), 1, 0, "out"))
            return
        # B: music + vocal on its drop (window end); its music rises over the last bar ("слишком резко")
        win.b(O, [(w, 0.0)] + ramp(win.end - 4, win.end, 0, 1, "in"))
        win.b(V, [(w, 0.0)] + ramp(win.end - 1, win.end, 0, 1))
        # A: music carries the energy to the end, drums last
        win.a(O, ramp(q(0.75), win.end, 1, 0.15, "out"))
        win.a(V, ramp(q(0.75) - 1, q(0.75), 1, 0, "out"))


class Mashup(Transition):
    name = "mashup"
    title = "Mashup"
    description = "A long section where A's vocal sings over B's full instrumental — a live bootleg."
    needs_stems = True
    min_bars = 8

    def bars(self, ctx):
        return 32 if ctx.mode == "full" else 24

    def score(self, ctx):
        if not (ctx.stems and ctx.can_beatmatch) or ctx.a_vocal_tail < 0.35 or ctx.key < 0.85:
            return 0
        # by ear: every mashup and acapella drop in the all-tracks set got 5/5 - a touch more often
        return 0.85 * ctx.key * (0.5 + ctx.a_vocal_tail) * _freshness(ctx, self.name) * ctx.stretch_penalty

    def handoff(self, w, length, params):
        # by ear ("слишком долго вокал Б шел", "вокал долго вступал"): B's singer takes over on a line
        # at 3/4, not with a slow fade over the last eighth
        return Handoff(w + length * 3 / 4, w + length * 5 / 8, w + length * 7 / 8)

    def apply(self, win: Window):
        w, f = win.w, win.frac
        win.b(GAIN, ramp(w, f(1 / 16), 0, 1, "in"))
        win.b(V, [(w, 0.0)] + ramp(f(3 / 4), f(3 / 4) + 1, 0, 1, "in"))
        win.b(EQ_LOW, [(w, 0.0)] + step(f(1 / 8), 0, 1))
        win.a(B_, step(f(1 / 8), 1, 0))
        win.a(D, step(f(1 / 8), 1, 0))
        win.a(O, ramp(w, f(1 / 8), 1, 0, "out"))
        win.a(V, ramp(f(3 / 4) - 1, f(3 / 4), 1, 0, "out"))


class DoubleDrop(Transition):
    name = "double_drop"
    title = "Double drop"
    description = ("Peak-time move: B comes in straight on its drop while A is still in its own; for a "
                   "phrase both drops ride together on one bassline, then A steps back.")
    needs_stems = True
    min_bars = 8

    def bars(self, ctx):
        return 16

    def entry(self, ctx):
        return "drop"

    def score(self, ctx):
        if not (ctx.stems and ctx.can_beatmatch) or ctx.rest or ctx.key < 0.85:
            return 0
        if min(ctx.a.energy_level, ctx.b.energy_level) < 0.7:
            return 0
        # a showpiece: rare, only between two compatible bangers
        return 0.75 * ctx.key * _freshness(ctx, self.name) * ctx.stretch_penalty

    def handoff(self, w, length, params):
        # A's singer ends its line around B's drop (within a bar), B's singer comes in on a line in the
        # second half, as the drops ride together first ("вокал обрезан": it used to be a fixed fade at W)
        return Handoff(w, w - 4, w + 4, b_min=w + length / 2)

    def apply(self, win: Window):
        w, f = win.w, win.frac
        # B hits at full force with its bassline; A keeps drums and music, gives up bass and vocal
        win.a(B_, step(w, 1, 0))
        win.a(V, ramp(w - 1, w, 1, 0, "out"))
        win.a(EQ_LOW, [(w, 1.0), (w + 0.01, 0.6)])
        win.b(V, [(w, 0.0)] + ramp(f(0.5) - 0.5, f(0.5), 0, 1, "in"))
        win.b(GAIN, [(w, 0.85)] + ramp(f(0.5), win.end, 0.85, 1))
        # second half: A steps back, its music first, drums on the last bar
        win.a(O, ramp(f(0.5), f(0.75), 1, 0, "out"))
        win.a(GAIN, ramp(f(0.5), win.end, 1, 0, "out"))


class AcapellaDrop(Transition):
    name = "acapella_drop"
    title = "Acapella into the drop"
    description = ("Bass-swap style blend, but on the last bar everything drops out except B's vocal; "
                   "the drop then hits in full.")
    needs_stems = True
    min_bars = 8

    def bars(self, ctx):
        return 8

    def score(self, ctx):
        if not (ctx.stems and ctx.can_beatmatch):
            return 0
        target = ctx.b.first_drop if ctx.mode == "full" else (ctx.b.highlights[0].start_bar if ctx.b.highlights else None)
        if not target or not ctx.b.bar_vocal or ctx.b.vocal_amount(target - 1, target) < 0.5:
            return 0  # needs B singing on the bar before its drop
        if ctx.key < 0.5:
            return 0  # 8 bars of both tracks' music: needs keys that agree
        lift = 1.0 if ctx.energy_delta >= 0 else 0.6
        return 0.95 * lift * (0.3 + 0.7 * ctx.key) * _freshness(ctx, self.name) * ctx.stretch_penalty

    def handoff(self, w, length, params):
        return self.mid_handoff(w, length)

    def apply(self, win: Window):
        w, end = win.w, win.end
        brk = end - 4  # the last bar of the overlap
        lands = win.tr.params.get("_lands", True)
        win.b(GAIN, ramp(w, win.frac(0.5), 0, 1, "in"))
        win.b(EQ_LOW, [(w, 0.0)] + step(win.frac(0.5), 0, 1))
        win.a(EQ_LOW, step(win.frac(0.5), 1, 0))
        win.a(GAIN, [(w, 1.0)] + ramp(brk - 0.5, brk, 1, 0, "out"))
        if not lands:
            return  # B's drop is not at the window end: plain blend
        for lane in (D, B_, O):
            win.b(lane, [(w, 1.0)] + step(brk, 1, 0) + step(end, 0, 1))
