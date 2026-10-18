"""SetList + analyses -> MixPlan (layout on the master beat grid, tempo curve, automation)."""

import numpy as np

from ..analysis import TrackAnalysis
from ..config import TARGET_LUFS
from ..engine.plan import (GAIN, PITCH, SEND_VECHO, SEND_VREVERB, SEND_VSLOW, SEND_VSWELL, STEM, VOCAL_DRY, VOCAL_FILTER, MixPlan,
                           SetList, Slot)
from ..transitions import Window, get
from ..transitions.base import ramp

RIDE_BEATS = 64  # tempo ride length after a beatmatched transition
PITCH_RETURN_BEATS = 32  # glide back to the original key after the mix-in
VOCAL = STEM + "vocals"
NO_RIDE = {"acapella_drop", "loop_roll", "riser"}  # transitions whose level dips / surges are deliberate


def build(sl: SetList, analyses: dict[str, TrackAnalysis]) -> MixPlan:
    slots: list[Slot] = []
    for i, item in enumerate(sl.items):
        an = analyses[item.track_id]
        beats = (item.end_bar - item.start_bar) * 4 * item.ratio
        start = 0.0
        if i:
            tr = sl.transitions[i - 1]
            overlap = tr.bars * 4
            prev = slots[-1]
            if overlap > min(prev.beats, beats):
                raise ValueError(f"transition {i} overlap {overlap} beats longer than a slot")
            start = prev.end - overlap
        slots.append(Slot(
            path=item.path, track_id=item.track_id, title=item.title,
            bpm=an.bpm * item.ratio, t0=an.t0, start=start, beats=beats,
            src_beat=item.start_bar * 4 * item.ratio, pitch_shift=item.pitch_shift,
            gain_db=float(np.clip(TARGET_LUFS - an.lufs, -12, 12)),
        ))

    tempo = [(0.0, slots[0].bpm)]
    for i, tr in enumerate(sl.transitions):
        a, b = slots[i], slots[i + 1]
        t = get(tr.type)
        w, end = b.start, b.start + max(tr.bars, 0) * 4
        if w - a.start < t.pre_bars * 4:
            raise ValueError(f"transition {i} ({tr.type}) needs {t.pre_bars} bars of track A before it")
        # where the next transition (if any) starts touching B
        if i + 1 < len(sl.transitions):
            nt = sl.transitions[i + 1]
            next_touch = b.end - max(nt.bars, 0) * 4 - get(nt.type).pre_bars * 4
        else:
            next_touch = b.end
        if t.beatmatched:
            # Both tracks meet at the geometric-mean tempo: A rides towards it before the overlap,
            # B starts there and rides to its own tempo afterwards (each stretched by half the gap).
            meet = a.bpm if abs(b.bpm / a.bpm - 1) < 0.03 else float(np.sqrt(a.bpm * b.bpm))
            if meet != a.bpm:
                ramp_start = max(tempo[-1][0] + 1e-3, w - RIDE_BEATS, a.start)
                tempo += [(ramp_start, a.bpm), (w, meet)]
            else:
                tempo.append((w, meet))
            tempo.append((end, meet))
            ride_end = max(end + 1e-3, min(end + RIDE_BEATS, next_touch))
            tempo.append((ride_end, b.bpm))
        elif t.tempo_bridge_beats and abs(b.bpm / a.bpm - 1) > 0.01:
            # e.g. a loop roll that speeds up into B's tempo: no hard jump
            tempo += [(max(tempo[-1][0] + 1e-3, w - t.tempo_bridge_beats), a.bpm), (w, b.bpm)]
        else:
            jump = min(a.end, w)  # for a gap the new tempo starts right at A's cut
            tempo += [(jump - 1e-3, a.bpm), (jump, b.bpm)]

    plan = MixPlan(slots=slots, tempo=tempo)
    for i, tr in enumerate(sl.transitions):
        get(tr.type).apply(Window(plan, i, i + 1, slots[i + 1].start, tr.bars * 4, tr))
        if tr.type in NO_RIDE or tr.bars <= 0:
            plan.fx.setdefault(str(i), {})["no_ride"] = True
    _pitch_returns(plan, sl)
    _vocal_handoffs(plan, sl, analyses)
    _cut_on_breath(plan, sl, analyses)
    _close_set(plan, sl, analyses)
    return plan


CUT_TYPES = {"cut"}


def _cut_on_breath(plan: MixPlan, sl: SetList, analyses: dict[str, TrackAnalysis]) -> None:
    """A cut lands on the one, but a rapper rarely stops there. Silence A's vocal stem at its last breath
    in the final bars (8th-note levels) so the line ends whole; the beat still plays to the cut."""
    for i, tr in enumerate(sl.transitions):
        if tr.type not in CUT_TYPES or tr.bars > 0:
            continue
        slot, an = plan.slots[i], analyses[plan.slots[i].track_id]
        if not an.vocal8 or not an.has_stems:
            continue
        cut = slot.end - get(tr.type).tail_beats
        if max(_vocal_db8(an, slot, cut - 0.5), _vocal_db8(an, slot, cut - 1.0)) < -15:
            continue  # already a pause before the cut
        # a breath: an 8th note clearly below the line around it (rap rarely goes fully silent)
        lv = [_vocal_db8(an, slot, cut - 0.5 * k) for k in range(1, 17)]
        floor = max(-15.0, float(np.median(lv)) - 6.0)
        # by ear: a vocal that vanishes bars before the cut sounds like a dropout ("вокал пропал и все
        # затихло"); only the last line's final breath, within a beat of the cut, is a clean stop
        for k in (1, 2):
            x = cut - 0.5 * k
            if lv[k - 1] < floor:
                plan.automate(i, f"{STEM}vocals", [(x - 0.01, 1.0)] + ramp(x, x + 0.25, 1, 0) + [(slot.end, 0.0)])
                plan.events.append({"beat": x, "text": f"{slot.title}: vocal stops on a breath {cut - x:g} beats before the cut"})
                break


def _close_set(plan: MixPlan, sl: SetList, analyses: dict[str, TrackAnalysis]) -> None:
    """If the closing track is cut before its natural end, fade it out over its last phrase instead of a stop."""
    i = len(plan.slots) - 1
    slot, item, an = plan.slots[i], sl.items[i], analyses[sl.items[i].track_id]
    if item.end_bar >= an.n_bars:
        return  # plays to the end of the file
    end = slot.end
    plan.automate(i, GAIN, ramp(end - 16, end, 1, 0, "out"))
    plan.events.append({"beat": end - 16, "text": f"set closes with a fade-out on {slot.title}"})


def _next_touch(plan: MixPlan, sl: SetList, i: int) -> float:
    """Global beat where slot i's outgoing transition starts to affect it."""
    slot = plan.slots[i]
    if i >= len(sl.transitions):
        return slot.end
    tr = sl.transitions[i]
    return slot.end - max(tr.bars, 0) * 4 - get(tr.type).pre_bars * 4


def _pitch_returns(plan: MixPlan, sl: SetList) -> None:
    """A harmonic shift only matters while the tracks overlap: glide back to the original key after."""
    for i, slot in enumerate(plan.slots):
        if not slot.pitch_shift or i == 0:
            continue
        mixed_in = slot.start + max(sl.transitions[i - 1].bars, 0) * 4
        back = min(mixed_in + PITCH_RETURN_BEATS, _next_touch(plan, sl, i))
        if back <= mixed_in + 4:
            continue  # no room to return; stay shifted
        plan.automate(i, PITCH, [(slot.start, slot.pitch_shift)] + ramp(mixed_in, back, slot.pitch_shift, 0))
        plan.events.append({"beat": mixed_in, "text": f"{slot.title}: key glides back {-slot.pitch_shift:+d} st to the original over {int((back - mixed_in) / 4)} bars"})


def _vocal_at(an: TrackAnalysis, slot: Slot, beat: float) -> float:
    if not an.bar_vocal:
        return 0.0
    ratio = slot.bpm / an.bpm
    bar = int((slot.src_beat + beat - slot.start) / (4 * ratio))
    return an.bar_vocal[bar] if 0 <= bar < len(an.bar_vocal) else 0.0


def _vocal_db8(an: TrackAnalysis, slot: Slot, beat: float) -> float:
    """Vocal level (dB) on the 8th note at a global beat."""
    ratio = slot.bpm / an.bpm
    track_beat = (slot.src_beat + beat - slot.start) / ratio
    return an.vocal_db8(int(np.floor(track_beat * 2)))


def _db8_floor(an: TrackAnalysis, slot: Slot, around: float) -> float:
    """Level below which an 8th note counts as a breath / gap, relative to the line around it."""
    lv = sorted(_vocal_db8(an, slot, around + 0.5 * k) for k in range(-8, 8))
    return max(-15.0, lv[len(lv) // 2] - 6.0)


def _eighth(an: TrackAnalysis, slot: Slot) -> float:
    """One 8th note of the track, in global beats (a quarter beat when it plays half-time)."""
    return 0.5 * slot.bpm / an.bpm


def _phrase_starts(an: TrackAnalysis, slot: Slot, lo: float, hi: float, rest: float) -> list[float]:
    """Line starts in [lo, hi] with at least `rest` beats of silence before them (a new phrase, not an
    onset after a quick breath)."""
    e = _eighth(an, slot)
    out = []
    for x in np.arange(lo, hi + 1e-6, e):
        floor = _db8_floor(an, slot, x)
        if _vocal_db8(an, slot, x) >= floor and all(_vocal_db8(an, slot, x - k * e) < floor
                                                     for k in range(1, int(round(rest / e)) + 1)):
            out.append(float(x))
    return out


def _line_start_after(an: TrackAnalysis, slot: Slot, h: float, reach: float = 4.0) -> float | None:
    """Earliest point >= h (within `reach` beats) where a new line begins: a gap, then singing."""
    floor, e = _db8_floor(an, slot, h), _eighth(an, slot)
    for x in np.arange(h, h + reach + 1e-6, e):
        if _vocal_db8(an, slot, x - e) < floor <= _vocal_db8(an, slot, x):
            return float(x)
    return None


def _line_ends(an: TrackAnalysis, slot: Slot, lo: float, hi: float) -> list[tuple[float, float]]:
    """Line ends (first quiet 8th after singing) in [lo, hi], each with the pause that follows it in beats
    (8 when the singer stays quiet for two bars or more)."""
    e = _eighth(an, slot)
    out = []
    for x in np.arange(lo, hi + 1e-6, e):
        if _vocal_db8(an, slot, x) < _db8_floor(an, slot, x) <= _vocal_db8(an, slot, x - e):
            nxt = _line_start_after(an, slot, x + e, reach=8.0)
            out.append((float(x), float(nxt - x) if nxt is not None else 8.0))
    return out


def _sings(an: TrackAnalysis, slot: Slot, lo: float, hi: float) -> bool:
    """Is there a lead vocal in [lo, hi]? Bar-level amount, or 8th notes near the singer's loud level
    (ad-libs and short hooks under the bar threshold still get chopped audibly - loop roll 'вокал обрывается')."""
    if max((_vocal_at(an, slot, x) for x in np.arange(lo, max(hi, lo + 1), 4.0)), default=0.0) >= 0.4:
        return True
    if not an.vocal8:
        return False
    lv = [_vocal_db8(an, slot, x) for x in np.arange(lo, max(hi, lo + 1), _eighth(an, slot))]
    return float(np.mean([v > -18 for v in lv])) >= 0.25


def _vocal_handoffs(plan: MixPlan, sl: SetList, analyses: dict[str, TrackAnalysis]) -> None:
    """Two lead vocals at once clash, and a voice faded slowly in or out mid-line sounds broken. Every
    transition that layers the decks names a handoff window (Transition.handoff): A's singer stops at the
    end of a line there - preferring a real pause between phrases over a quick breath - and B's singer
    comes in at the start of its next line with a short fade."""
    last_slowed = -99
    for i, tr in enumerate(sl.transitions):
        mode = tr.params.get("vocals", "handoff")
        a, b = plan.slots[i], plan.slots[i + 1]
        w, length = b.start, max(tr.bars, 0) * 4
        spec = get(tr.type).handoff(w, length, tr.params)
        if spec is None or mode == "both":
            continue
        an_a, an_b = analyses[a.track_id], analyses[b.track_id]
        if not (an_a.has_stems and an_b.has_stems and an_a.bar_vocal and an_b.bar_vocal):
            continue
        end = w + length
        lo, hi = max(spec.lo, a.start + 1), min(spec.hi, a.end - 0.5)
        if hi < lo:
            continue
        target = float(np.clip(spec.target, lo, hi))
        a_sings = mode != "a" and _sings(an_a, a, lo, min(max(hi, end), a.end))
        b_line = spec.b == "line" and _sings(an_b, b, max(w, lo), end + 4)
        if not (a_sings or b_line):
            continue
        b_from = max(w, lo if spec.b_min is None else spec.b_min)
        if mode == "a":
            b_from = max(b_from, end)
        ea, eb = _eighth(an_a, a), _eighth(an_b, b)
        if a_sings:
            if mode == "b":
                stop, out = w, ramp(w - 0.5, w, 1, 0, "out")
                release = stop
            else:
                # a real line end has a pause of at least a beat of the track after it; by ear a quick breath
                # inside a verse that raps straight through is still a cut ("вокал обрезан", "выход обрезан")
                real = lambda cs: [c for c in cs if c[1] >= 2 * ea - 1e-6]
                ends = real(_line_ends(an_a, a, lo, hi))
                if not ends:
                    # look a little wider for a pause between lines
                    span = hi - lo
                    wide = real(_line_ends(an_a, a, max(a.start + 1, lo - span / 2), min(a.end - 0.5, hi + span / 4)))
                    if wide:
                        ends, lo, hi = wide, min(lo, wide[0][0]), max(hi, wide[-1][0])
                span = max(hi - lo, 4.0)
                if ends:
                    def cost(c):
                        x, pause = c
                        k = round((x - w) / 4)
                        off_bar = abs((x - w) / 4 - k) * 4  # rap lines end just before a downbeat
                        phrase = 0 if k % 4 == 0 else 0.1 if k % 2 == 0 else 0.2
                        # a longer pause between lines is worth moving a few beats for
                        return abs(x - target) / span + 0.1 * min(off_bar, 2) + phrase - 0.3 * min(pause, 4) / 4

                    stop, pause = min(ends, key=cost)
                    release = stop
                    # the voice's own tail (reverb, breath) decays over up to a beat, not chopped on an 8th
                    out = [(stop - 0.01, 1.0)] + ramp(stop, stop + float(np.clip(pause, 0.25, 2 * ea)), 1, 0, "out", 8)
                else:
                    stop, out, hi, release = _exit_without_pause(plan, i, tr, an_a, a, an_b, b, w, target, lo, hi,
                                                                 span, ea, real, i - last_slowed)
                    if any(e["beat"] == stop and "slowed exit" in e["text"] for e in plan.events):
                        last_slowed = i
            plan.clear(i, VOCAL, lo - 1, max(hi, end) + 1)
            plan.automate(i, VOCAL, [(min(lo, out[0][0]) - 0.5, 1.0)] + out)
            b_from = max(b_from, release)
            if not any(e["beat"] == stop and e["text"].startswith(f"{a.title}: no pause") for e in plan.events):
                plan.events.append({"beat": stop, "text": f"{a.title}: vocal ends its line ({(stop - w) / 4:+.1f} bars)"})
        if b_line:
            # by ear ("резкое вступление вокала"): B came in on any onset after a quick breath - often mid-verse -
            # at full level after an 8th. Now: a real phrase start (a beat of silence before it) within two bars,
            # opened half a beat early so the pickup / breath before the first word is in
            starts = _phrase_starts(an_b, b, b_from, b_from + 8, 2 * eb)
            if starts:
                b_in = starts[0]
                up = ramp(b_in - 2 * eb, b_in, 0, 1, "in", 8)
                how = "on a phrase"
            elif _vocal_db8(an_b, b, b_from) < _db8_floor(an_b, b, b_from) and _line_start_after(an_b, b, b_from, 8.0) is None:
                up = [(b_from, 0.0), (b_from + 0.01, 1.0)]  # quiet for two bars: whatever comes next is a fresh line
                how = "after a rest"
            else:
                # raps straight through: in on a downbeat, the first word grows out of a reverse-reverb swell
                # (the mirror of A's exit into a reverb)
                x = w + 4 * np.ceil((b_from - w) / 4 - 1e-6)
                up = ramp(x - 2 * eb, x, 0, 1, "in", 8)
                plan.clear(i + 1, SEND_VSWELL, x - 1, x + 3)
                plan.automate(i + 1, SEND_VSWELL, [(x - 0.01, 0.0), (x, 1.0), (x + 1.5, 1.0), (x + 1.51, 0.0)])
                how = "on a downbeat out of a reverse-reverb swell"
            plan.clear(i + 1, VOCAL, w, max(end, up[-1][0]))
            plan.automate(i + 1, VOCAL, [(w, 0.0)] + up)
            plan.events.append({"beat": up[-1][0], "text": f"{b.title}: vocal comes in {how} ({(up[-1][0] - w) / 4:+.1f} bars)"})


SLOWED_SPACING = 4  # transitions between two slowed exits: an effect that is heard too often becomes a gimmick


def _auto_exit(an: TrackAnalysis, a: Slot, an_b: TrackAnalysis, b: Slot, stop: float, since_slowed: int) -> str:
    """slowed (the voice sinks) where the set does not lift - B no more energetic than A, or a slower / half-time
    B - and B leaves room for the tail (no singing in the beat and a half after the stop); reverb (the voice
    goes up into the air) otherwise: energy up, B singing straight away."""
    calmer = an_b.energy_level - an.energy_level <= 0 or b.bpm < a.bpm * 0.97 or b.bpm / an_b.bpm < a.bpm / an.bpm
    floor = _db8_floor(an_b, b, stop)
    room = all(_vocal_db8(an_b, b, stop + x) < floor for x in np.arange(0, 1.5, _eighth(an_b, b)))
    return "slowed" if calmer and room and since_slowed > SLOWED_SPACING else "reverb"


def _exit_without_pause(plan: MixPlan, i: int, tr, an: TrackAnalysis, a: Slot, an_b: TrackAnalysis, b: Slot,
                        w: float, target: float, lo: float, hi: float, span: float, ea: float, real,
                        since_slowed: int = 99):
    """A raps straight through the handoff window. tr.params["vocal_exit"] picks how its voice leaves:
      finish - raps on over B's instrumental to its next real line end (until a bar before A leaves), B waits;
               without one, falls back to auto
      reverb - stops on a phrase-grid downbeat, the last syllable rings out in a vocal-only reverb
      slowed - a beat before the downbeat the voice slows down a little (1 -> 0.75, pitch falls) into a
               reverb (user's idea: "slowed и reverb"); the dry voice hands over to the slowed copy there
      echo   - the same stop, the last syllable repeats in a vocal-only tempo echo (3/8)
      filter - the voice runs on for one bar through a closing high-pass, then is gone
      cut    - a dry stop on the downbeat
      auto   - (default) slowed or reverb from the tracks, see _auto_exit
    Returns (stop, vocal-lane points, new hi, beat from which B may sing)."""
    # by ear: of five styles on 3-4 / 5-6 the vocal reverb won, then a soft slowed was liked too ("вроде
    # норм, может оставить и то и это"): both stay, picked per transition
    style = tr.params.get("vocal_exit", "auto")
    if style == "finish":
        far = real(_line_ends(an, a, hi, a.end - 4))  # up to a bar before A leaves
        if far:
            stop, pause = far[0]
            out = [(stop - 0.01, 1.0)] + ramp(stop, stop + float(np.clip(pause, 0.25, 2 * ea)), 1, 0, "out", 8)
            plan.events.append({"beat": stop, "text": f"{a.title}: raps on over B to the end of its line"})
            return stop, out, max(hi, stop), stop
        style = "auto"
    # a decisive stop on a downbeat of the phrase grid ("вокал четко обрезан" was liked)
    bars = [w + 4 * k for k in range(int(np.ceil((lo - w) / 4)), int(np.floor((hi - w) / 4)) + 1)]
    grid = lambda x: 0 if round((x - w) / 4) % 4 == 0 else 0.1 if round((x - w) / 4) % 2 == 0 else 0.2
    stop = float(min(bars, key=lambda x: abs(x - target) / span + grid(x))) if bars else target
    if style == "auto":
        style = _auto_exit(an, a, an_b, b, stop, since_slowed)
    fade = max(ea, 0.25)
    out = [(stop - 0.01, 1.0)] + ramp(stop, stop + fade, 1, 0, "out", 8)
    release = stop
    last_word = [(stop - 1.5, 0.0)] + ramp(stop - 1.5, stop - 0.25, 0, 1) + [(stop + fade, 1.0), (stop + fade + 0.01, 0.0)]
    if style == "reverb":
        plan.automate(i, SEND_VREVERB, last_word)
    elif style == "echo":
        plan.automate(i, SEND_VECHO, last_word)
    elif style == "slowed":
        s0 = stop - 1  # the slowdown takes the last beat of the line and stretches it past the downbeat
        out = [(s0 - 0.01, 1.0), (s0 + 0.02, 0.0)]
        plan.automate(i, SEND_VSLOW, [(s0 - 0.03, 0.0), (s0, 1.0), (s0 + 2.5, 1.0), (s0 + 2.51, 0.0)])
        release = stop + 1
    elif style == "filter":
        sweep = 4.0
        plan.automate(i, VOCAL_DRY, [(stop - 0.01, 0.0), (stop, 1.0)] + ramp(stop + sweep - 1, stop + sweep, 1, 0))
        plan.automate(i, VOCAL_FILTER, ramp(stop, stop + sweep, 0.25, 0.95, "exp"))
        release = stop + sweep
    plan.events.append({"beat": stop, "text": f"{a.title}: no pause in the line - {style} exit on the downbeat"})
    return stop, out, hi, release
