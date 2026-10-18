"""Tracks -> SetList: order, tempo ratios, harmonic shifts, transition choice and play regions."""

from dataclasses import dataclass, field

import numpy as np

from ..analysis import TrackAnalysis
from ..analysis.key import shift_camelot
from ..engine.plan import SetItem, SetList, SetTransition
from ..transitions import REGISTRY, Ctx, get
from .compat import best_ratio, harmonic
from .order import order_tracks


@dataclass
class PlanOptions:
    mode: str = "full"  # full | highlights
    keep_order: bool = False
    duration_min: float | None = None
    transitions: list[str] | None = None  # allowed transition types (None = all)
    forced: list[str] | None = None  # exact transition per pair, cycled
    hl_bars: int = 32
    seed: int = 0
    max_shift: int = 1  # max semitones of harmonic pitch shift (0 = off)
    energy_curve: str = "peak"
    program: str | None = None  # e.g. "club": pick tracks from the crate along a section plan
    sections: list[str] | None = None  # pre-assigned section per track (keeps a previous selection)
    use_stems: bool = True
    log: object = field(default=lambda *a: None)


def _hl(an: TrackAnalysis):
    return an.highlights[0]


def _region_hint(an: TrackAnalysis, mode: str, hl_bars: int) -> tuple[int, int]:
    """Rough region (before transitions are known), used for vocal / intro context."""
    if mode == "highlights":
        h = _hl(an)
        return h.start_bar, min(an.n_bars, h.start_bar + hl_bars)
    return 0, an.n_bars


def _exit_bars(a: TrackAnalysis, opts: PlanOptions) -> tuple[int, int]:
    """Bars of A that will sound under the next track (rough, before regions are fixed)."""
    end = a.active_end if opts.mode == "full" else _region_hint(a, opts.mode, opts.hl_bars)[1]
    return max(0, end - 16), end


def _entry_bars(b: TrackAnalysis, opts: PlanOptions) -> tuple[int, int]:
    start = b.active_start if opts.mode == "full" else max(0, _hl(b).start_bar - 16)
    return start, start + 16


def _sings_early(b: TrackAnalysis, opts: PlanOptions) -> bool:
    """B's vocal is audible during the overlap or the 8-bar key glide right after it."""
    s0, s1 = _entry_bars(b, opts)
    return b.vocal_amount(s0, s1 + 16) > 0.3


def _section_start_before(an: TrackAnalysis, bar: int, reach: int = 16) -> int:
    """Nearest section start at or before `bar` (within `reach` bars), else `bar` on the 4-bar grid:
    a track should come in at the top of a section, not halfway through a chorus."""
    starts = [sg.start_bar for sg in an.segments if bar - reach <= sg.start_bar <= bar]
    return max(starts) if starts else max(0, bar // 4 * 4)


def _energetic_start(an: TrackAnalysis, hs: int) -> int:
    """For a cut-in: the earliest section start up to 16 bars before the highlight that already plays
    with most of its energy (B lands with drums on the one, not in a quiet intro), else the highlight."""
    ref = an.bar_energy[min(hs, an.n_bars - 1)]
    for sg in an.segments:
        head = an.bar_energy[sg.start_bar:sg.start_bar + 4]
        if hs - 16 <= sg.start_bar < hs and len(head) and min(head) >= 0.75 * ref:
            return sg.start_bar
    return hs


def _ctx(a, b, key, gap, opts: PlanOptions, recent) -> Ctx:
    a_s, a_e = _region_hint(a, opts.mode, opts.hl_bars)
    b_s, _ = _region_hint(b, opts.mode, opts.hl_bars)
    if opts.mode == "full":
        intro = b.segments[0] if b.segments else None
        clean = bool(intro and intro.label in ("intro", "groove") and intro.energy < 0.6 and intro.bars >= 8)
        b_head = b.vocal_amount(0, 16)
        a_tail = a.vocal_amount(max(0, (a.outro_start or a.n_bars) - 32), a.outro_start or a.n_bars)
    else:
        seg = b.segment_at(max(0, b_s - 8))
        clean = bool(seg and seg.energy < 0.6 and b_s >= 8)
        b_head = b.vocal_amount(max(0, b_s - 16), b_s + 8)
        a_tail = a.vocal_amount(a_s, a_e)
    return Ctx(a=a, b=b, key=key, bpm_gap=gap, energy_delta=b.energy_level - a.energy_level,
               stems=opts.use_stems and a.has_stems and b.has_stems, a_vocal_tail=a_tail,
               b_vocal_head=b_head, b_clean_intro=clean, mode=opts.mode, recent=list(recent))


def explain(name: str, ctx: Ctx, bars: float, shift: int, r: float, lands: bool = False) -> str:
    """Why this technique, in DJ terms."""
    a, b = ctx.a, ctx.b
    t = get(name)
    tempo = f"{a.bpm:g}->{b.bpm * r:g} BPM" + (f" (half-time of {b.bpm:g})" if r == 0.5 else (f" (double-time of {b.bpm:g})" if r == 2 else ""))
    parts = []
    if ctx.bpm_gap < 0.03:
        parts.append(f"{tempo}, locked")
    elif t.beatmatched:
        parts.append(f"{tempo}: both decks meet in the middle (~{ctx.bpm_gap / 2:.0%} each)")
    elif ctx.can_beatmatch:
        parts.append(f"{tempo}: blendable, but a clean break on the phrase is the stronger move")
    elif name == "loop_roll":
        parts.append(f"{tempo}: the roll accelerates into the new tempo")
    elif name == "reverb_wash":
        parts.append(f"{tempo} is too far to blend: A washes out in reverb on its phrase end, B rises from the tail")
    else:
        parts.append(f"{tempo} is too far to blend")
    if t.harmonic_overlap and bars > 0:
        k = "compatible" if ctx.key >= 0.8 else ("workable" if ctx.key >= 0.5 else "clashing")
        kb = shift_camelot(b.camelot, shift)
        parts.append(f"keys {a.camelot}/{kb} {k}" + (f" (B {b.camelot} shifted {shift:+d} st for the overlap)" if shift else ""))
    elif bars > 0 and not t.harmonic_overlap:
        parts.append(f"keys {a.camelot}/{b.camelot}: overlap kept percussive, music changes on B's drop")
    if bars > 0 and lands:
        parts.append(f"{bars:g}-bar overlap over B's lead-in, so B's drop lands as A ends")
    if ctx.energy_delta > 0.1:
        parts.append("energy up")
    elif ctx.energy_delta < -0.1:
        parts.append("energy down")
    return "; ".join(parts)


def choose_transition(ctx: Ctx, opts: PlanOptions, rng: np.random.Generator) -> tuple[str, str]:
    allowed = opts.transitions or list(REGISTRY)
    scored = []
    for name in allowed:
        t = get(name)
        if t.needs_stems and not ctx.stems:
            continue
        if t.beatmatched and not ctx.can_beatmatch:
            continue
        if t.bars(ctx) > 0 and t.min_bars > ctx.room:
            continue  # would be squeezed out of shape
        s = t.score(ctx)
        if s > 0:
            scored.append((s, name))
    if not scored:
        fallback = next((n for n in ("reverb_wash", "loop_roll", "cut") if n in allowed), allowed[0])
        return fallback, "fallback"
    # small seeded jitter only breaks near-ties; the best-scoring technique wins
    scored = sorted(((s * (1 + 0.04 * rng.standard_normal()), n) for s, n in scored), reverse=True)
    reason = ", ".join(f"{n}={s:.2f}" for s, n in scored[:4])
    return scored[0][1], reason


def _phrase_points(an: TrackAnalysis) -> list[int]:
    """Bars on an 8-bar phrase grid anchored at every section start."""
    pts = {an.n_bars}
    for seg in an.segments:
        pts.update(range(seg.start_bar, an.n_bars + 1, 8))
    return sorted(pts)


def _snap_down(an: TrackAnalysis, bar: int) -> int:
    return max((p for p in _phrase_points(an) if p <= bar), default=bar)


def _snap_up(an: TrackAnalysis, bar: int) -> int:
    return min((p for p in _phrase_points(an) if p >= bar), default=bar)


def _quiet_exit(an: TrackAnalysis, earliest: int, latest: int, tail: int = 0, caught: bool = False) -> int:
    """For cut-style exits: the latest phrase boundary where the singer is not mid-line.
    8-bar phrase points are preferred; 4-bar sub-phrases are allowed (e.g. inside an outro)."""
    phrase8 = set(_phrase_points(an))
    cands = phrase8 | {p for p in range(0, an.n_bars + 1, 4)}
    best, best_cost = latest, None
    for p in sorted(cands):
        if not (max(earliest, latest - 32) <= p <= latest) or p < 2:
            continue
        # mid-line = singing on the last beat before the cut AND straight after it
        end = 4 * p + tail
        before = max(an.vocal_on(end - 1), 0.5 * an.vocal_on(end - 2)) * (0.5 if caught else 1.0)
        br = an.breath_before(end, bars=1)
        if br is not None and br <= 1.0:  # a breath on the last beat: the line ends with the cut
            before = min(before, 0.1)
        # a cut leaves from a full part of A (the end of a hook), never from a thinning tail
        ref = float(np.median(an.bar_energy))
        thin = max(0.0, 0.8 * ref - an.bar_energy[min(max(p - 1, 0), an.n_bars - 1)])
        after = an.vocal_on(end)
        # chorus-to-verse: leaving at the end of a section is what the listener expects
        section_end = any(abs(p - sg.end_bar) <= 1 for sg in an.segments)
        cost = (before * (0.4 + 0.6 * after) + 0.15 * (latest - p) / 8 + (0 if p in phrase8 else 0.15)
                + (0 if section_end else 0.2) + 2.0 * thin)
        if best_cost is None or cost < best_cost:
            best, best_cost = p, cost
    return best


def _region(an: TrackAnalysis, opts: PlanOptions, entry: str | None, l_in: int, l_out: int,
            pre_out: int, overlapping_exit: bool, last: bool, tail: int = 0,
            caught: bool = False) -> tuple[int, int]:
    """Play region in the track's own bars. l_in/l_out/pre_out are in track bars."""
    n = an.n_bars
    # never mix out of a drum-less tail tag; skip leading silence (snap both to the 4-bar grid)
    act_end = _snap_down(an, an.active_end) if an.active_end < n else n
    act_start = _snap_up(an, an.active_start) if an.active_start > 0 else 0
    if opts.mode == "full":
        drop = an.first_drop or 0
        s = drop if entry == "drop" else act_start
        e = act_end if not last else n
        outro = an.outro_start
        if not last and not overlapping_exit and outro and n - outro >= 8 and outro - s >= 32:
            e = outro
        if not last and not overlapping_exit and an.bar_vocal:
            # may reach into the outro while the drums still play
            e = _quiet_exit(an, s + 16 + pre_out, max(e, min(an.active_end, n) // 4 * 4), tail, caught)
    else:
        h = _hl(an)
        hs, he = h.start_bar, min(n, h.start_bar + opts.hl_bars)
        if entry == "drop":
            s = hs
        elif entry == "cut":
            s = _energetic_start(an, hs)
        else:
            lead = l_in if entry is not None and l_in > 0 else 8  # a cut-in still gets a lead-in phrase
            # hs - lead is on the phrase grid of the highlight, so B's drop lands as A leaves; snap to a
            # section start only when one is close (a longer quiet lead-in would play alone after A)
            s = _section_start_before(an, max(0, hs - lead), reach=2)
        e = min(n, he + (l_out if overlapping_exit else 0) + (8 if last else 0))
        if not last:
            # never into the fade-out: past the active end only to finish the highlight itself
            e = min(e, act_end if act_end - s >= 16 + l_in else max(act_end, min(he, h.end_bar)))
            if not overlapping_exit and an.bar_vocal:
                e = _quiet_exit(an, max(s + 8 + pre_out, he - 8), min(n, an.active_end // 4 * 4, he + 8), tail, caught)
        if e - s < min(n, opts.hl_bars) and entry == "cut" and not last:
            e = min(n, act_end, s + opts.hl_bars)  # a cut-in keeps its energetic start: play on instead
            if not overlapping_exit and an.bar_vocal:
                e = _quiet_exit(an, max(s + 8 + pre_out, e - 8), min(n, an.active_end // 4 * 4, e + 8), tail, caught)
        if e - s < min(n, opts.hl_bars) and entry != "cut":
            # a highlight near the end of the file: start earlier rather than air a 30-second snippet
            s = _section_start_before(an, max(0, int(e - min(n, opts.hl_bars))))
    if overlapping_exit and not last and l_out:
        drops = [sg.start_bar for sg in an.segments
                 if sg.label == "drop" and e - l_out <= sg.start_bar < e and sg.start_bar - s >= l_in + l_out + 16]
        if drops:  # mix out over the build / break before it instead
            e = drops[0]
    need = l_in + max(l_out, pre_out) + 8
    if e - s < need:
        e = min(n, s + need)
    if e - s < need:
        s = max(0, e - need)
    return s, e


def _protect_climax(a: TrackAnalysis, r: float, bars: float) -> float:
    """A short track often saves its biggest moment (bass in, full drop) for the end. The outgoing
    overlap must not bury it: leave at least 8 bars of the last drop solo before mixing B in."""
    end = a.active_end
    drops = [sg.start_bar for sg in a.segments if sg.label == "drop" and sg.start_bar < end - 4]
    if not drops or drops[-1] < end - 32:
        return bars
    room = (end - drops[-1] - 8) * r
    if room >= bars:
        return bars
    return max(4, int(room) // 4 * 4)


def _room(a: TrackAnalysis, b: TrackAnalysis, ra: float, rb: float, opts: PlanOptions) -> float:
    """Longest overlap (output bars) before it buries A's closing drop or runs past B's lead-in."""
    # full tracks: never bury the closing drop under B. Highlights: the region leaves before a drop of A that
    # would fall inside the overlap (see _region), so the blend keeps its length instead of shrinking to 4 bars
    room = _protect_climax(a, ra, 1e9) if opts.mode == "full" else 1e9
    target = b.first_drop if opts.mode == "full" else _hl(b).start_bar
    lead = (target or 0) - (b.active_start if b.bar_stem_db else 0)
    if lead >= 8:
        room = min(room, int(lead * rb) // 4 * 4)
    return room


def plan_set(tracks: list[TrackAnalysis], opts: PlanOptions) -> SetList:
    rng = np.random.default_rng(opts.seed)
    sections: list[str] = []
    if opts.sections:
        seq, sections = list(tracks), list(opts.sections)
    elif opts.program:
        from .program import select_and_order

        picked = select_and_order(tracks, opts.program, opts.duration_min or 60, opts.mode, opts.hl_bars,
                                  opts.max_shift, log=opts.log)
        seq, sections = [t for t, _ in picked], [s for _, s in picked]
    else:
        order = list(range(len(tracks))) if opts.keep_order else order_tracks(tracks, opts.energy_curve, opts.max_shift)
        seq = [tracks[i] for i in order]
    kinds = {}
    if opts.program:
        from .program import section_kinds

        kinds = section_kinds(opts.program)

    # Walk the chain: tempo ratio, transition choice, then a harmonic shift only if the transition
    # actually layers the two tracks (a pitch shift buys nothing on a cut and costs audio quality).
    ratios, shifts = [1.0], [0]
    cams = [seq[0].camelot]
    trans: list[SetTransition] = []
    recent: list[str] = []
    for i in range(len(seq) - 1):
        prev, cur = seq[i], seq[i + 1]
        r, gap = best_ratio(prev.bpm * ratios[-1], cur.bpm)
        s_best, k_best = harmonic(prev, cur, opts.max_shift, _exit_bars(prev, opts), _entry_bars(cur, opts))
        if s_best and _sings_early(cur, opts):
            # a shifted vocal and the glide back to its own key sound like a warped tape: keep the
            # singer in key and let the transition choice handle a clash (percussive overlap instead)
            s_best, k_best = harmonic(prev, cur, 0, _exit_bars(prev, opts), _entry_bars(cur, opts))
        ctx = _ctx(prev, cur, k_best, gap, opts, recent)
        if sections:
            ctx.rest = kinds.get(sections[i + 1]) == "rest"
        ctx.room = _room(prev, cur, ratios[-1], r, opts)
        ctx.groove_change = abs(r - ratios[-1]) > 1e-6
        forced = opts.forced[i % len(opts.forced)] if opts.forced else "auto"
        if forced != "auto":
            name, reason = forced, "forced"
        else:
            name, reason = choose_transition(ctx, opts, rng)
        t = get(name)
        if t.needs_stems and not ctx.stems:
            name, reason = "bass_swap" if ctx.can_beatmatch else "reverb_wash", f"{name} needs stems"
            t = get(name)
        if t.beatmatched and not ctx.can_beatmatch:
            name, reason = "reverb_wash", f"{name} needs beatmatching (gap {ctx.bpm_gap:.0%})"
            t = get(name)
        bars = t.bars(ctx)
        if bars > 0 and not ctx.groove_change and r < 1 and name != "mashup":
            # bars() counts the tracks' own bars, the overlap is laid out on the master grid: a run of ~130 BPM
            # tracks played at ratio 0.5 sits on a 65 BPM grid, where 16 bars were a minute of overlap (v14:
            # 4.1★ vs 4.5★, "вокал стрем"). Only downwards: slow tracks doubled to ratio 2 would get a minute of
            # their own bars. A mashup stays long on purpose (user: "mashup оставь длинным")
            bars = max(4, int(round(bars * r / 4)) * 4)
        entry = t.entry(ctx)
        lands = False
        if bars > 0 and entry == "intro":
            # B's drop / highlight should land as A finishes: the overlap lives in B's lead-in.
            target = cur.first_drop if opts.mode == "full" else _hl(cur).start_bar
            lead = (target or 0) - (cur.active_start if cur.bar_stem_db else 0)
            if lead >= 8:  # a 4-bar lead-in would squeeze the blend into ~7 s: overlap the drop instead
                bars = max(4, min(bars, int(lead * r) // 4 * 4))
                lands = bars == int(lead * r) // 4 * 4
        if bars > 0 and t.harmonic_overlap and ctx.groove_change:
            # a half/double-time change (either way): 16 output bars are 30 s or more of two songs (and their
            # vocals) on top of each other; the old check only caught ratio 2 and let 0.5 through
            bars = min(bars, 8)
        if bars > 0:
            bars = max(4, min(bars, ctx.room))
        shift = s_best if bars > 0 and t.harmonic_overlap else 0
        ratios.append(r)
        shifts.append(shift)
        cams.append(cur.camelot)  # the shift glides back after the overlap (see builder)
        params = {"entry": entry, "_key": round(float(k_best), 2), "_lands": lands}
        why = explain(name, ctx, bars, shift, r, lands)
        if reason in ("forced", "fallback") or "needs" in reason:
            why = f"{why} [{reason}]"
        trans.append(SetTransition(type=name, bars=bars, reason=why, params=params))
        recent.append(name)

    items = []
    for i, an in enumerate(seq):
        tin = trans[i - 1] if i else None
        tout = trans[i] if i < len(trans) else None
        r = ratios[i]
        to_src = lambda bars: int(np.ceil(bars / r))  # output bars -> track bars
        entry = tin.params["entry"] if tin else None
        l_in = to_src(max(tin.bars, 0)) if tin else 0
        l_out = to_src(max(tout.bars, 0)) if tout else 0
        pre_out = to_src(get(tout.type).pre_bars) if tout else 0
        tail = get(tout.type).tail_beats if tout else 0
        caught = bool(tout and get(tout.type).fx_tail)
        s, e = _region(an, opts, entry, l_in, l_out, pre_out, bool(tout and tout.bars > 0), tout is None, tail, caught)
        if tail and e + tail / 4 / r <= an.n_bars:
            e = e + tail / 4 / r
        if tout is None and e >= an.n_bars:
            # the closing track plays out to the end of the file: keep its natural ending / decay
            e = max(e, int((an.duration - an.t0) / an.bar_len * 16) / 16)
        items.append(SetItem(path=an.path, track_id=an.track_id, title=an.title, start_bar=s,
                             end_bar=e, ratio=r, pitch_shift=shifts[i], section=sections[i] if sections else ""))

    # Shrink overlaps that do not fit inside the regions.
    for i, tr in enumerate(trans):
        a, b = items[i], items[i + 1]
        room = min((a.end_bar - a.start_bar) * a.ratio, (b.end_bar - b.start_bar) * b.ratio) - 4
        while tr.bars > 4 and tr.bars > room:
            tr.bars -= 4
            tr.params["_lands"] = False  # B's region was laid out for the longer overlap: its drop no longer lands at the end
        if tr.type == "drum_bed":
            # is B sung over the bars it actually enters on? (in highlights mode that is not its intro: the
            # old active_start check sent bars with no vocal into the sung-intro branch)
            s0 = int(b.start_bar)
            tr.params["_b_intro_vocal"] = round(seq[i + 1].vocal_amount(s0, s0 + int(np.ceil(tr.bars / b.ratio))), 2)

    sl = SetList(items=items, transitions=trans)
    if opts.duration_min and not (opts.program or opts.sections):
        sl = _fit_duration(sl, tracks, opts)
    return sl


def set_duration(sl: SetList, analyses: dict[str, TrackAnalysis]) -> float:
    total = 0.0
    for i, it in enumerate(sl.items):
        bpm = analyses[it.track_id].bpm * it.ratio
        beats = (it.end_bar - it.start_bar) * 4 * it.ratio
        if i:
            beats -= sl.transitions[i - 1].bars * 4
        total += beats * 60 / bpm
    return total


def _fit_duration(sl: SetList, tracks: list[TrackAnalysis], opts: PlanOptions) -> SetList:
    analyses = {t.track_id: t for t in tracks}
    target = opts.duration_min * 60
    if set_duration(sl, analyses) <= target * 1.05:
        return sl
    if opts.mode == "full":
        opts.log("set too long for full tracks -> highlights mode")
        opts.mode = "highlights"
        opts.duration_min = None
        sl = plan_set(tracks, opts)
        opts.duration_min = target / 60
        if set_duration(sl, analyses) <= target * 1.05:
            return sl
    while len(sl.items) > 2 and set_duration(sl, analyses) > target * 1.05:
        sl.items.pop()
        sl.transitions.pop()
    return sl
