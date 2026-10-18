"""Mix critic: per-transition QA checks of a planned and rendered mix."""

from dataclasses import dataclass, field

import numpy as np

from .analysis import TrackAnalysis
from .analysis.key import chroma_fit, key_compat, shift_camelot
from .engine.plan import EQ_LOW, GAIN, MixPlan, SetList
from .engine.tempo import TempoMap
from .transitions import get


def _lane_at(plan: MixPlan, slot: int, lane: str, beat: float, default: float) -> float:
    pts = plan.lanes.get(f"{slot}:{lane}")
    if not pts:
        return default
    return float(np.interp(beat, [p[0] for p in pts], [p[1] for p in pts]))


def _src_bar(an: TrackAnalysis, slot, beat: float) -> float:
    ratio = slot.bpm / an.bpm
    return (slot.src_beat + beat - slot.start) / (4 * ratio)


def _phrase_err(an: TrackAnalysis, bar: float) -> float:
    """Distance (bars) from the nearest point on an 8-bar grid anchored at a section boundary."""
    bounds = [s.start_bar for s in an.segments] + [an.n_bars]
    best = 99.0
    for b0 in bounds:
        d = (bar - b0) % 8
        best = min(best, d, 8 - d)
    return best


@dataclass
class TransitionReport:
    index: int
    type: str
    scores: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return float(np.mean(list(self.scores.values()))) if self.scores else 1.0


def critique(sl: SetList, plan: MixPlan, analyses: dict[str, TrackAnalysis], audio: np.ndarray | None = None,
             sr: int = 44100, probes: dict | None = None) -> list[TransitionReport]:
    """`probes`: overlap excerpts collected by `render(..., probes=...)`; saves re-rendering slots."""
    tm = TempoMap(plan.tempo, plan.total_beats)
    out = []
    for i, tr in enumerate(sl.transitions):
        a, b = plan.slots[i], plan.slots[i + 1]
        an_a, an_b = analyses[a.track_id], analyses[b.track_id]
        rep = TransitionReport(i, tr.type)
        t = get(tr.type)
        w = b.start
        L = max(tr.bars, 0) * 4
        a_out_bar = _src_bar(an_a, a, a.end)
        a_win_bar = _src_bar(an_a, a, max(w, a.start))
        b_in_bar = _src_bar(an_b, b, w)

        # phrase alignment: A's exit / window start and B's entry on the phrase grid
        a_phrase_bar = a_out_bar - t.tail_beats / 4  # the extra "on the one" beat is part of the move
        errs = [_phrase_err(an_a, a_phrase_bar), _phrase_err(an_b, b_in_bar)]
        if L:
            errs.append(_phrase_err(an_a, a_win_bar))
        worst = max(errs)
        on4 = all(min(x % 4, 4 - x % 4) < 0.5 for x in (a_phrase_bar, b_in_bar))
        rep.scores["phrase"] = 1.0 if worst < 0.5 else (0.75 if on4 and worst <= 4 else float(np.clip(1 - worst / 4, 0, 1)))
        if worst >= 1:
            rep.notes.append(f"off-phrase by {max(errs):.0f} bars (A out bar {a_out_bar:.0f}, B in bar {b_in_bar:.0f})")

        if L:
            beats = np.arange(w, w + L, 1.0)
            ea = np.array([an_a.bar_energy[min(int(_src_bar(an_a, a, x)), an_a.n_bars - 1)] for x in beats])
            eb = np.array([an_b.bar_energy[min(int(_src_bar(an_b, b, x)), an_b.n_bars - 1)] for x in beats])
            ga = np.array([_lane_at(plan, i, GAIN, x, 1) for x in beats])
            gb = np.array([_lane_at(plan, i + 1, GAIN, x, 1) for x in beats])
            # how much of each track is actually playing (a drums-only bed is not a full drop)
            mix = {"stem:drums": 0.3, "stem:bass": 0.25, "stem:other": 0.25, "stem:vocals": 0.2}
            fa = np.array([sum(w * _lane_at(plan, i, k, x, 1) for k, w in mix.items()) for x in beats])
            fb = np.array([sum(w * _lane_at(plan, i + 1, k, x, 1) for k, w in mix.items()) for x in beats])
            busy = float(np.mean((ea * ga * fa > 0.6) & (eb * gb * fb > 0.6)))
            rep.scores["busy"] = 1 - busy
            if busy > 0.25:
                rep.notes.append(f"drop-on-drop for {busy:.0%} of the overlap")

            # vocals (after stem automation)
            if an_a.bar_vocal and an_b.bar_vocal:
                va = np.array([an_a.bar_vocal[min(int(_src_bar(an_a, a, x)), an_a.n_bars - 1)] for x in beats])
                vb = np.array([an_b.bar_vocal[min(int(_src_bar(an_b, b, x)), an_b.n_bars - 1)] for x in beats])
                sa = np.array([_lane_at(plan, i, "stem:vocals", x, 1) for x in beats]) * ga
                sb = np.array([_lane_at(plan, i + 1, "stem:vocals", x, 1) for x in beats]) * gb
                clash = float(np.mean((va * sa > 0.4) & (vb * sb > 0.4)))
                rep.scores["vocal_clash"] = 1 - clash
                if clash > 0.1:
                    rep.notes.append(f"two vocals at once for {clash:.0%} of the overlap")

            la = np.array([_lane_at(plan, i, EQ_LOW, x, 1) * _lane_at(plan, i, "stem:bass", x, 1) for x in beats]) * ga
            lb = np.array([_lane_at(plan, i + 1, EQ_LOW, x, 1) * _lane_at(plan, i + 1, "stem:bass", x, 1) for x in beats]) * gb
            bass = float(np.mean((la > 0.5) & (lb > 0.5)))
            rep.scores["bass_clash"] = 1 - min(1, bass * 4)
            if bass > 0.05:
                rep.notes.append(f"two basslines for {bass:.0%} of the overlap")

            if t.harmonic_overlap:
                cam_a = shift_camelot(an_a.camelot, a.pitch_shift)
                cam_b = shift_camelot(an_b.camelot, b.pitch_shift)
                k = key_compat(cam_a, cam_b)
                # measured notes of the overlapping parts, when available (robust to key mix-ups)
                bars_a = (int(_src_bar(an_a, a, w)), int(_src_bar(an_a, a, w + L)) + 1)
                bars_b = (int(_src_bar(an_b, b, w)), int(_src_bar(an_b, b, w + L)) + 1)
                pa, pb = an_a.chroma(*bars_a), an_b.chroma(*bars_b)
                if pa is not None and pb is not None:
                    k = 0.4 * k + 0.6 * chroma_fit(pa, pb, b.pitch_shift)  # A has glided back by its exit
                rep.scores["key"] = k
                if k < 0.6:
                    rep.notes.append(f"key clash {cam_a} vs {cam_b} ({k:.2f})")

        # tempo: stretch beyond 6 % starts to smear; hard jumps are a last resort
        stretch_a = abs(tm.bpm_at(a.end - 0.01) / a.bpm - 1)
        stretch_b = abs(tm.bpm_at(w + 0.01) / b.bpm - 1)
        st = max(stretch_a, stretch_b)
        rep.scores["tempo"] = float(np.clip(1 - max(0, st - 0.05) / 0.12, 0, 1))
        if t.tempo_bridge_beats and not t.beatmatched:
            # the stretch happens on a repeating loop, where it is inaudible; only a smooth ramp remains
            rep.scores["tempo"] = 0.9
        elif not t.beatmatched:
            jump = abs(b.bpm / a.bpm - 1)
            if jump > 0.03:
                rep.scores["tempo"] *= 0.7
                rep.notes.append(f"hard tempo jump {a.bpm:g} -> {b.bpm:g}")

        # a cut / echo must not chop a vocal line mid-word
        if L == 0 and an_a.bar_vocal:
            e_beat = int(round(a_out_bar * 4))
            caught = 0.5 if t.fx_tail else 1.0  # an echo / reverb tail carries the word out
            v_end = (max(an_a.vocal_on(e_beat - 1), 0.5 * an_a.vocal_on(e_beat - 2)) * caught
                     * (0.4 + 0.6 * an_a.vocal_on(e_beat)))
            # what is still audible over the last 8th note (a vocal stopped on a breath right before counts)
            v_gain = min(_lane_at(plan, i, "stem:vocals", x, 1) * _lane_at(plan, i, GAIN, x, 1)
                         for x in (a.end - 0.5, a.end - 0.2))
            cut = v_end * v_gain
            rep.scores["vocal_cut"] = float(np.clip(1.2 - cut, 0, 1))
            if cut > 0.5:
                rep.notes.append(f"A's vocal is cut mid-line (activity {v_end:.1f} across the cut)")

        if audio is not None and t.beatmatched and L >= 8:
            rep.scores["sync"] = _sync_score(plan, i, tm, rep, (probes or {}).get(i))

        if audio is not None:
            rep.scores["loudness"] = _loudness_score(audio, sr, tm.time(w), tm.time(max(a.end, w + L)), rep)
        out.append(rep)
    return out


def _sync_score(plan: MixPlan, i: int, tm: TempoMap, rep, probe=None) -> float:
    """Compare the kick onsets of both slots of a beatmatched overlap, each rendered on its own."""
    import librosa

    from .engine.render import render_slot

    sr = 44100
    w, e = plan.slots[i + 1].start, plan.slots[i].end
    kicks = []
    for side, k in enumerate((i, i + 1)):
        if probe is not None and probe[side] is not None:
            seg = probe[side]
        else:  # render just the overlap of this slot
            t0, t1 = tm.time(w), tm.time(e)
            x, off = render_slot(k, plan.slots[k], plan, tm, window=(t0, t1))
            seg = x[:, int(t0 * sr) - off:int(t1 * sr) - off].mean(0)
        S = np.abs(librosa.stft(seg, n_fft=1024, hop_length=64))
        f = librosa.fft_frequencies(sr=sr, n_fft=1024)
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[f < 150]), sr=sr, hop_length=64)
        kicks.append(librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=64, units="time"))
    ka, kb = kicks
    if len(ka) < 4 or len(kb) < 4:
        return 1.0
    d = np.array([kb[np.abs(kb - t).argmin()] - t for t in ka])
    d = d[np.abs(d) < 0.06]
    if len(d) < 4:
        return 1.0
    off_ms = abs(float(np.median(d))) * 1000
    if off_ms > 8:
        rep.notes.append(f"kicks off by {off_ms:.0f} ms in the overlap")
    return float(np.clip(1 - max(0, off_ms - 5) / 20, 0, 1))


def _st_loudness(audio: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    seg = audio[:, max(0, int(t0 * sr)):int(t1 * sr)].mean(0)
    hop = int(0.4 * sr)
    win = int(3 * sr)
    vals = []
    for s in range(0, max(1, len(seg) - win), hop):
        vals.append(10 * np.log10(np.mean(seg[s:s + win] ** 2) + 1e-10))
    return np.array(vals)


def _loudness_score(audio, sr, t0, t1, rep) -> float:
    ref = np.median(np.concatenate([_st_loudness(audio, sr, t0 - 30, t0 - 2), _st_loudness(audio, sr, t1 + 2, t1 + 30)]))
    during = _st_loudness(audio, sr, t0 - 2, t1 + 4)
    if not len(during):
        return 1.0
    bump = during.max() - ref
    hole = ref - during.min()
    if bump > 2:
        rep.notes.append(f"loudness bump +{bump:.1f} dB")
    if hole > 6:
        rep.notes.append(f"loudness hole -{hole:.1f} dB")
    return float(np.clip(1 - max(0, bump - 1.5) / 4 - max(0, hole - 5) / 10, 0, 1))


def summary(reports: list[TransitionReport]) -> str:
    lines = []
    for r in reports:
        sc = " ".join(f"{k}={v:.2f}" for k, v in r.scores.items())
        lines.append(f"#{r.index} {r.type:11s} total={r.total:.2f}  {sc}")
        for n in r.notes:
            lines.append(f"      - {n}")
    if reports:
        lines.append(f"SET SCORE {np.mean([r.total for r in reports]):.3f}  (min {min(r.total for r in reports):.2f})")
    return "\n".join(lines)
