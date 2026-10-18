"""Offline renderer: MixPlan -> stereo audio."""

from functools import lru_cache

import numpy as np
from pedalboard import Compressor, Limiter, Pedalboard, time_stretch

from .. import stems as stemlib
from ..audio import db_to_amp, load
from ..config import SR, STEM_NAMES
from . import dsp
from .plan import (EQ_HIGH, EQ_LOW, EQ_MID, FILTER, GAIN, PITCH, SEND_REVERB, SEND_VREVERB, SEND_VECHO, SEND_VSLOW, SEND_VSWELL, VOCAL_BUSES, VOCAL_DRY, VOCAL_FILTER,
                   STEM, MixPlan,
                   Slot, neutral)
from .tempo import TempoMap

TAIL_S = 8.0
JOIN_FADE = int(0.003 * SR)


@lru_cache(maxsize=8)
def _full(path: str) -> np.ndarray:
    return load(path, SR)


@lru_cache(maxsize=8)
def _stems(tid: str) -> dict[str, np.ndarray]:
    return stemlib.load_stems(tid)


def _excerpt(x: np.ndarray, start: int, n: int) -> np.ndarray:
    out = np.zeros((x.shape[0], n), np.float32)
    s0, s1 = max(start, 0), min(start + n, x.shape[1])
    if s1 > s0:
        out[:, s0 - start:s1 - start] = x[:, s0:s1]
    return out


def _virtual_source(x: np.ndarray, slot: Slot) -> np.ndarray:
    """Concatenate the slot's source segments (loops / rolls / edits) on the source beat grid."""
    d = 60.0 / slot.bpm * SR  # samples per source beat
    parts = []
    prev_end_src = None
    for seg in slot.default_segments():
        n = int(round((seg.out + seg.length) * d)) - int(round(seg.out * d))
        start = int(round(slot.t0 * SR + seg.src * d))
        part = _excerpt(x, start, n)
        if prev_end_src is not None and abs(prev_end_src - start) > 1 and n > 2 * JOIN_FADE:
            r = np.linspace(0, 1, JOIN_FADE, dtype=np.float32)
            part[:, :JOIN_FADE] *= r
            parts[-1][:, -JOIN_FADE:] *= r[::-1]
        parts.append(part)
        prev_end_src = start + n
    return np.concatenate(parts, axis=1)


def _env(points, beats: np.ndarray, lane: str):
    if not points:
        return neutral(lane)
    xs = np.array([p[0] for p in points])
    vs = np.array([p[1] for p in points])
    return np.interp(beats, xs, vs).astype(np.float32)


CHUNK_PAD = int(0.3 * SR)
CHUNK_XF = int(0.010 * SR)  # half crossfade: neighbouring chunks overlap by 2 * CHUNK_XF
CHUNK_ALIGN = int(0.0015 * SR)  # max phase-alignment shift at a seam


def _stretch(x: np.ndarray, slot: Slot, tm: TempoMap, n_out: int, pitch=None,
             lags: list[int] | None = None, hq: bool = True,
             active: tuple[float, float] | None = None) -> np.ndarray:
    """Time-stretch the virtual source onto the master tempo curve.

    Rubber Band drifts slightly over long buffers (~0.1-0.2 %), which is audible when beatmatching,
    so we stretch bar-sized chunks and pin each one to its exact position on the master grid.
    `pitch` is an optional automation lane (global beat -> semitones), applied per chunk.
    `lags`: seam alignment shifts. Pass an empty list to record them, or a filled one to reuse them,
    so a stem stretched with the full track's lags stays sample-aligned with it.
    `hq=False`: Rubber Band's fast mode (~5x faster, for drafts). `active`: (b0, b1) slot-relative beats;
    chunks outside it are left silent (partial renders). Silent input chunks are never stretched.
    """
    reuse = bool(lags)
    seam = 0
    if pitch:
        pb = np.array([q[0] for q in pitch])
        pv = np.array([q[1] for q in pitch])
        pitch_at = lambda beats: np.interp(beats, pb, pv)
    else:
        pitch_at = lambda beats: np.full(np.shape(beats), float(slot.pitch_shift))
    d = 60.0 / slot.bpm * SR
    t_start = tm.time(slot.start)
    # Chunk boundaries a bit after each downbeat (away from the kick transient); where the master
    # tempo is ramping, split down to single beats so the per-chunk constant factor stays accurate.
    bounds = [0.0]
    for bar in range(int(np.ceil(slot.beats / 4))):
        probe = slot.start + np.linspace(bar * 4, bar * 4 + 4, 9)
        ramping = np.ptp(tm.bpm_at(probe)) > 0.05 or np.ptp(pitch_at(probe)) > 0.02
        for k in (range(4) if ramping else (0,)):
            b = bar * 4 + k + 0.3
            if bounds[-1] < b < slot.beats and not (bar == 0 and k == 0):
                bounds.append(b)
    bounds.append(slot.beats)
    xf = CHUNK_XF
    y = np.zeros((x.shape[0], n_out + 2 * xf), np.float32)
    fade = np.linspace(0, 1, 2 * xf, dtype=np.float32)
    prev_tail = None  # raw (unfaded) audio of the previous chunk over the coming seam
    for b0, b1 in zip(bounds, bounds[1:]):
        if active is not None and (b1 <= active[0] or b0 >= active[1]):
            if lags is not None and not reuse:
                lags.append(0)
            seam += 1
            prev_tail = None
            continue
        s0, s1 = int(round(b0 * d)), int(round(b1 * d))
        o0 = int(round((tm.time(slot.start + b0) - t_start) * SR))
        o1 = int(round((tm.time(slot.start + b1) - t_start) * SR))
        want = o1 - o0
        f = float(tm.bpm_at(slot.start + (b0 + b1) / 2)) / slot.bpm
        semis = float(pitch_at(slot.start + (b0 + b1) / 2))
        lo = max(0, s0 - CHUNK_PAD)
        seg = x[:, lo:s1 + CHUNK_PAD]
        exact = abs(f - 1) < 1e-4 and abs(semis) < 1e-3
        if not np.any(seg):  # silence (e.g. a send that is closed here): nothing to stretch
            if lags is not None and not reuse:
                lags.append(0)
            seam += 1
            prev_tail = None
            continue
        if exact:
            out, lead = seg, s0 - lo
        else:
            out = time_stretch(seg, SR, f, semis, high_quality=hq)
            lead = int(round((s0 - lo) / f))
        a = lead - xf
        # WSOLA-style seam: nudge this chunk (+-2 ms) so its head lines up in phase with the
        # previous chunk's tail; avoids comb dips / bumps on sustained notes and vocals.
        if reuse:
            a += lags[seam] if seam < len(lags) else 0
        elif prev_tail is not None and not exact:
            ref = prev_tail.mean(0)
            best, best_lag = -np.inf, 0
            for lag in range(-CHUNK_ALIGN, CHUNK_ALIGN + 1, 4):
                st = a + lag
                if st < 0 or st + 2 * xf > out.shape[1]:
                    continue
                c = float(np.dot(ref, out[:, st:st + 2 * xf].mean(0)))
                if c > best:
                    best, best_lag = c, lag
            a += best_lag
            if lags is not None:
                lags.append(best_lag)
        elif lags is not None:
            lags.append(0)
        seam += 1
        n_piece = want + 2 * xf
        piece = np.zeros((x.shape[0], n_piece), np.float32)
        src0, src1 = max(a, 0), min(a + n_piece, out.shape[1])
        if src1 > src0:
            piece[:, src0 - a:src1 - a] = out[:, src0:src1]
        prev_tail = piece[:, -2 * xf:].copy()
        if b0 > 0:
            piece[:, :2 * xf] *= fade
        if b1 < slot.beats:
            piece[:, -2 * xf:] *= fade[::-1]
        # piece starts at output o0 - xf, i.e. index o0 in the xf-padded buffer
        end = min(o0 + n_piece, y.shape[1])
        y[:, o0:end] += piece[:, :end - o0]
    return np.ascontiguousarray(y[:, xf:xf + n_out])


WINDOW_MARGIN = 2.0  # s rendered around a partial-render window (filter / crossfade warm-up)


def render_slot(i: int, slot: Slot, plan: MixPlan, tm: TempoMap, draft: bool = False,
                window: tuple[float, float] | None = None) -> tuple[np.ndarray, int]:
    """Render one slot; returns (audio incl. FX tail, start sample on the master).

    `window`: (t0, t1) master seconds - render only that part (plus a small margin); the result is
    cropped to it and the offset moved accordingly."""
    lanes = plan.lanes_for(i)
    stem_lanes = sorted(l for l in lanes if l.startswith(STEM))
    full = _full(slot.path)

    t_start, t_end = tm.time(slot.start), tm.time(slot.end)
    offset = int(round(t_start * SR))
    n_out = int(round(t_end * SR)) - offset
    active, c0, c1 = None, 0, n_out
    if window is not None:
        lo, hi = window[0] - WINDOW_MARGIN, window[1] + WINDOW_MARGIN
        # chunk seam alignment is a chain (each chunk phase-locks to the previous one): start stretching
        # four bars early so it has converged to the full render's by the window
        active = (float(tm.beat(lo)) - slot.start - 16, float(tm.beat(hi)) - slot.start)
        c0 = int(np.clip(round((lo - t_start) * SR), 0, n_out))
        c1 = int(np.clip(round((hi - t_start) * SR), c0, n_out))

    # The original always plays; a stem lane only *subtracts* the part of that stem it turns down:
    #   src = full + sum(stem * (env - 1))
    # mixed at the source, before time-stretching, so the stretcher sees one coherent signal: no
    # phasey residue from cancelling separately stretched stems, and wherever every lane is at 1
    # (before / after the transition) the output is the untouched original.
    src = _virtual_source(full, slot)
    has = stemlib.has_stems(slot.track_id)
    d = 60.0 / slot.bpm * SR
    src_beats = slot.start + np.arange(src.shape[1]) / d  # virtual-source sample -> global beat
    stems = _stems(slot.track_id) if has and (stem_lanes or SEND_REVERB in lanes or any(l in lanes for l in VOCAL_BUSES)) else None
    if stems is not None and stem_lanes:
        for lane in stem_lanes:
            name = lane[len(STEM):]
            if name not in STEM_NAMES:
                continue
            delta = _env(lanes[lane], src_beats, lane) - 1.0
            if np.any(np.abs(delta) > 1e-4):
                src = src + _virtual_source(stems[name], slot) * delta
    hq = not draft
    y = _stretch(src, slot, tm, n_out, lanes.get(PITCH), None, hq, active)[:, c0:c1]
    wash = None
    if stems is not None and SEND_REVERB in lanes:
        # the reverb send takes the music only: a singer's last word hanging in the tail sounds like a
        # vocal throw (not wanted); the dry vocal still ends with the track
        voc = _virtual_source(stems["vocals"], slot)
        if f"{STEM}vocals" in lanes:
            voc = voc * _env(lanes[f"{STEM}vocals"], src_beats, f"{STEM}vocals")
        on = [b for b, v in lanes[SEND_REVERB] if v > 0]
        b0 = (min(on) if on else slot.start) - 1.0 - slot.start
        rng = (max(b0, active[0]), active[1]) if active is not None else (b0, slot.beats)
        wash = _stretch(src - voc, slot, tm, int(round(tm.time(slot.end) * SR)) - int(round(tm.time(slot.start) * SR)),
                        lanes.get(PITCH), None, hq, rng)[:, c0:c1]
        del voc
    # vocal-only buses (a line cut without a pause): each takes the raw vocal stem (before its own stem
    # mute) through its send envelope, so a reverb / echo tail or a filtered copy starts where the voice stops
    vbus = {}
    if stems is not None:
        for bus in VOCAL_BUSES:
            if bus not in lanes:
                continue
            on = [b for b, v in lanes[bus] if v > 0]
            b0 = (min(on) if on else slot.start) - 1.0 - slot.start
            b1 = (max(on) if on else slot.end) + 1.0 - slot.start
            rng = (max(b0, active[0]), min(b1, active[1])) if active is not None else (b0, min(b1, slot.beats))
            if rng[1] > rng[0]:
                vsend = _virtual_source(stems["vocals"], slot) * _env(lanes[bus], src_beats, bus)
                vbus[bus] = _stretch(vsend, slot, tm, int(round(tm.time(slot.end) * SR)) - int(round(tm.time(slot.start) * SR)),
                                     lanes.get(PITCH), None, hq, rng)[:, c0:c1]
                del vsend
    del src
    n_out, offset = c1 - c0, offset + c0
    t_start = offset / SR
    beats = tm.beat(t_start + np.arange(n_out) / SR)

    def tone(z):
        if any(l in lanes for l in (EQ_LOW, EQ_MID, EQ_HIGH)):
            z = dsp.eq3(z, SR, *(_env(lanes.get(l), beats, l) for l in (EQ_LOW, EQ_MID, EQ_HIGH)))
        if FILTER in lanes:
            z = dsp.dj_filter(z, SR, _env(lanes[FILTER], beats, FILTER))
        return z

    y = tone(y)
    if wash is not None:
        wash = tone(wash)
    vbus = {k: tone(v) for k, v in vbus.items()}
    if VOCAL_DRY in vbus and VOCAL_FILTER in lanes:
        vbus[VOCAL_DRY] = dsp.dj_filter(vbus[VOCAL_DRY], SR, _env(lanes[VOCAL_FILTER], beats, VOCAL_FILTER))

    y = dsp.fade_edges(y, SR)
    tail = int(TAIL_S * SR)
    out = np.zeros((2, n_out + tail), np.float32)
    level = db_to_amp(slot.gain_db)
    settings = plan.fx.get(str(i), {})
    if SEND_REVERB in lanes:
        send = np.pad((y if wash is None else wash) * _env(lanes[SEND_REVERB], beats, SEND_REVERB),
                      ((0, 0), (0, tail)))
        out += dsp.reverb(send, SR, settings.get("reverb_size", 0.9)) * level * 0.8
    for bus, z in vbus.items():
        z = np.pad(z * _env(lanes.get(GAIN), beats, GAIN), ((0, 0), (0, tail)))
        if bus == SEND_VREVERB:
            out += dsp.reverb(z, SR, settings.get("vreverb_size", 0.6)) * level * 0.7
        elif bus == SEND_VECHO:
            delay = settings.get("vecho_beats", 0.75) * 60.0 / float(tm.bpm_at(slot.end))
            out += dsp.echo(z, SR, delay, settings.get("vecho_feedback", 0.5)) * level * 0.8
        elif bus == SEND_VSWELL:
            # reverse reverb: the reverb of B's first word, played backwards so it swells up to the word
            on = [b for b, v in lanes[SEND_VSWELL] if v > 0]
            if on:
                a0 = int(round((tm.time(min(on)) - t_start) * SR))
                m = int(round((tm.time(min(on) + 2) - tm.time(min(on))) * SR))  # two beats of swell
                if 0 < a0 < z.shape[1]:
                    wet = dsp.reverb(z[:, a0:a0 + 3 * m], SR, settings.get("vswell_size", 0.8))[:, :m]
                    m = min(wet.shape[1], a0)
                    rev = wet[:, :m][:, ::-1] * np.linspace(0, 1, m, dtype=np.float32) ** 1.5
                    out[:, a0 - m:a0] += rev * level * 0.8
        elif bus == SEND_VSLOW:
            # from the send's start the voice plays on at a falling speed (1 -> 1 - depth over two beats, pitch
            # dropping with it), then fades; dry plus a reverb of the slowed voice ("slowed + reverb").
            # By ear 1 -> 0.5 was too much ("поменьше замедление"): 1 -> 0.75
            on = [b for b, v in lanes[SEND_VSLOW] if v > 0]
            if on:
                s0 = min(on)
                a0 = int(round((tm.time(s0) - t_start) * SR))
                n = int(round((tm.time(s0 + 2) - tm.time(s0)) * SR))
                if 0 <= a0 < z.shape[1] and n > 0:
                    k = np.arange(min(n, z.shape[1] - a0))
                    depth = settings.get("vslow_depth", 0.25)
                    pos = a0 + k - 0.5 * depth * k**2 / n  # speed 1 - depth * k / n
                    idx = np.arange(z.shape[1])
                    slow = np.zeros_like(z)
                    slow[:, a0:a0 + len(k)] = np.stack([np.interp(pos, idx, ch) for ch in z]) * (1 - k / n) ** 0.7
                    out += (slow * 0.85 + dsp.reverb(slow, SR, settings.get("vreverb_size", 0.6)) * 0.7) * level
        else:
            out += z * level
    out[:, :n_out] += y * _env(lanes.get(GAIN), beats, GAIN) * level
    return out, offset


def _master_fx(master: np.ndarray, plan: MixPlan, tm: TempoMap) -> None:
    def put(x, at):
        s0, end = max(at, 0), min(master.shape[1], at + x.shape[1])
        if end > s0:
            master[:, s0:end] += x[:, s0 - at:end - at]

    for fx in plan.master_fx:
        if fx["type"] == "riser":
            a, b = (int(round(tm.time(fx[k]) * SR)) for k in ("start", "end"))
            put(dsp.noise_riser(b - a, SR) * fx.get("gain", 0.25), a)
        elif fx["type"] == "roll":
            hit = dsp.clap(SR) * fx.get("gain", 0.3)
            start, end = fx["start"], fx["end"]
            beat = start
            while beat < end:
                prog = (beat - start) / (end - start)
                put(hit * (0.4 + 0.6 * prog), int(round(tm.time(beat) * SR)))
                beat += 1.0 if prog < 0.5 else (0.5 if prog < 0.75 else (0.25 if prog < 0.9 else 0.125))
        elif fx["type"] == "impact":
            put(dsp.impact(SR) * fx.get("gain", 0.5), int(round(tm.time(fx["at"]) * SR)))


RIDE_WIN = 3.0  # s, short-term loudness window
RIDE_HOP = 0.5
RIDE_MAX_CUT, RIDE_MAX_BOOST = 4.0, 3.0  # dB


def _short_term_db(x: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Short-term level (dB) of master[start:end] every RIDE_HOP s; returns (times, db)."""
    win, hop = int(RIDE_WIN * SR), int(RIDE_HOP * SR)
    ts, vals = [], []
    for a in range(max(0, start), max(start + 1, end - win), hop):
        seg = x[:, a:a + win]
        ts.append((a + win / 2) / SR)
        vals.append(10 * np.log10(float(np.mean(seg.astype(np.float64) ** 2)) + 1e-10))
    return np.array(ts), np.array(vals)


def _ride_transitions(master: np.ndarray, plan: MixPlan, tm: TempoMap) -> None:
    """Gain riding, like a DJ on the master fader: across each overlap, steer the combined level from
    A's level to B's, pulling down the bump of two full tracks and lifting the hole when both are
    filtered or thinned out. Only acts inside transitions (+ a couple of seconds)."""
    pad = 2.0
    for i in range(len(plan.slots) - 1):
        if plan.fx.get(str(i), {}).get("no_ride"):
            continue  # the transition's dynamics are the point (a break, a wash)
        a, b = plan.slots[i], plan.slots[i + 1]
        t0, t1 = tm.time(b.start), tm.time(max(a.end, b.start))
        if t1 - t0 < 2.0:
            continue
        # reference levels: A alone just before, B alone just after the overlap
        _, la = _short_term_db(master, int((t0 - 20) * SR), int((t0 - 1) * SR))
        _, lb = _short_term_db(master, int((t1 + 1) * SR), int((t1 + 20) * SR))
        if len(la) < 3 or len(lb) < 3:
            continue
        ref_a, ref_b = np.median(la), np.median(lb)
        ts, lv = _short_term_db(master, int((t0 - pad) * SR), int((t1 + pad) * SR))
        if len(ts) < 3:
            continue
        frac = np.clip((ts - t0) / (t1 - t0), 0, 1)
        target = ref_a + (ref_b - ref_a) * frac
        corr = np.clip(target - lv, -RIDE_MAX_CUT, RIDE_MAX_BOOST)
        corr[np.abs(corr) < 1.0] = 0.0  # leave small wobbles alone
        corr = np.convolve(corr, np.ones(5) / 5, mode="same")  # ~2.5 s smoothing
        taper = np.clip(np.minimum(ts - (t0 - pad), (t1 + pad) - ts) / pad, 0, 1)
        corr *= taper
        if not np.any(np.abs(corr) > 0.3):
            continue
        s0, s1 = int((t0 - pad) * SR), min(master.shape[1], int((t1 + pad) * SR))
        s0 = max(0, s0)
        n = s1 - s0
        g = 10 ** (np.interp(s0 / SR + np.arange(n) / SR, ts, corr, left=0, right=0) / 20)
        master[:, s0:s1] *= g.astype(np.float32)


PROBE_MIN_BEATS = 8  # overlaps the critic measures (kick sync)


def _overlap_probes(i: int, plan: MixPlan, tm: TempoMap, out: np.ndarray, offset: int) -> dict:
    """Mono excerpts of slot i over its overlaps with its neighbours, keyed by transition index, so the
    critic can check kick sync without rendering the slot again."""
    probes = {}
    for k in (i - 1, i):  # transition k is between slots k and k+1
        if k < 0 or k + 1 >= len(plan.slots):
            continue
        w, e = plan.slots[k + 1].start, plan.slots[k].end
        if e - w < PROBE_MIN_BEATS:
            continue
        a, b = int(tm.time(w) * SR) - offset, int(tm.time(e) * SR) - offset
        probes[k] = out[:, max(a, 0):max(b, 0)].mean(0)
    return probes


def _render_one(args):
    i, plan, draft = args
    tm = TempoMap(plan.tempo, plan.total_beats)
    out, offset = render_slot(i, plan.slots[i], plan, tm, draft)
    return out, offset, _overlap_probes(i, plan, tm, out, offset)


def render(plan: MixPlan, log=lambda *a: None, workers: int | None = None, draft: bool = False,
           probes: dict | None = None) -> np.ndarray:
    """Render the whole mix. `probes`: if given, filled with {transition: (A excerpt, B excerpt)} of
    every overlap of >= PROBE_MIN_BEATS beats (for the critic's kick-sync check)."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    tm = TempoMap(plan.tempo, plan.total_beats)
    total = int((tm.time(plan.total_beats) + TAIL_S) * SR)
    master = np.zeros((2, total), np.float32)
    workers = workers or min(len(plan.slots), max(1, (os.cpu_count() or 2) - 2))
    if workers > 1 and len(plan.slots) > 2:
        # slots are independent: render them in parallel processes, mixing each in as it arrives
        # (holding every rendered slot at once does not fit in memory for multi-hour sets)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = ex.map(_render_one, [(i, plan, draft) for i in range(len(plan.slots))])
            for i, (out, offset, pr) in enumerate(results):
                end = min(total, offset + out.shape[1])
                master[:, offset:end] += out[:, :end - offset]
                del out
                _collect(probes, i, pr)
    else:
        for i, s in enumerate(plan.slots):
            out, offset = render_slot(i, s, plan, tm, draft)
            end = min(total, offset + out.shape[1])
            master[:, offset:end] += out[:, :end - offset]
            _collect(probes, i, _overlap_probes(i, plan, tm, out, offset))
    _ride_transitions(master, plan, tm)
    _master_fx(master, plan, tm)
    # glue compressor + limiter in blocks, in place (a stateful board streams across block edges)
    board = Pedalboard([Compressor(threshold_db=-16, ratio=1.8, attack_ms=25, release_ms=250),
                        Limiter(threshold_db=-1.5, release_ms=120)])
    block = 30 * SR
    for a in range(0, total, block):
        b = min(total, a + block)
        master[:, a:b] = board.process(master[:, a:b], SR, reset=(a == 0))
    # trim trailing silence
    nz = np.flatnonzero(np.abs(master[:, -60 * SR:]).max(axis=0) > 1e-4)
    if len(nz):
        master = master[:, : max(0, total - 60 * SR) + nz[-1] + SR // 2]
    # The limiter lets fast transients through; scale (never clip) to -1 dBFS so lossy encoders
    # do not overshoot.
    peak = max(float(np.abs(master[:, a:a + block]).max()) for a in range(0, master.shape[1], block))
    if peak > 0.89:
        for a in range(0, master.shape[1], block):
            master[:, a:a + block] *= 0.89 / peak
    return master


def _collect(probes: dict | None, i: int, pr: dict) -> None:
    """Slot i's excerpt over transition k is A's side when k == i, B's side when k == i - 1."""
    if probes is None:
        return
    for k, seg in pr.items():
        probes.setdefault(k, [None, None])[0 if k == i else 1] = seg
