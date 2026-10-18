import numpy as np
import pytest

from neurojockey.engine.plan import SetItem, SetList, SetTransition, Slot
from neurojockey.engine.render import _stretch, render
from neurojockey.engine.tempo import TempoMap
from neurojockey.planner import PlanOptions, build, plan_set
from neurojockey.transitions import REGISTRY

SR = 44100


def test_tempo_map_roundtrip():
    tm = TempoMap([(0, 120), (64, 120), (128, 130)], 200)
    assert tm.time(64) == pytest.approx(32.0, abs=1e-6)
    for b in (10.0, 70.5, 150.0):
        assert tm.beat(tm.time(b)) == pytest.approx(b, abs=1e-6)


@pytest.mark.parametrize("master", [128, 124, 132])
def test_stretch_stays_on_grid(master):
    bpm = 128
    d = 60 / bpm * SR
    x = np.zeros((2, int(d * 64) + SR), np.float32)
    for k in range(64):
        x[:, int(k * d):int(k * d) + 64] = np.hanning(128)[64:]
    slot = Slot("p", "t", "t", bpm, 0.0, 0.0, 60, 0.0)
    tm = TempoMap([(0, master), (30, master), (60, master + 2)], 100)
    y = _stretch(x, slot, tm, int(round(tm.time(60) * SR)))
    env = np.abs(y[0])
    hits, i = [], 0
    while i < len(env):
        if env[i] > 0.3:
            hits.append(i)
            i += int(SR * 0.2)
        else:
            i += 1
    expected = np.array([tm.time(k) * SR for k in range(len(hits))])
    err_ms = np.abs(np.array(hits) - expected) / SR * 1000
    assert len(hits) >= 58
    assert err_ms.max() < 4.0  # <= 1.5 ms of it is the deliberate WSOLA seam alignment


def test_planner_orders_and_respects_tempo(demo_tracks):
    sl = plan_set(demo_tracks, PlanOptions(mode="highlights", seed=0))
    assert len(sl.items) == 4 and len(sl.transitions) == 3
    for tr, a, b in zip(sl.transitions, sl.items, sl.items[1:]):
        ta = next(t for t in demo_tracks if t.track_id == a.track_id)
        tb = next(t for t in demo_tracks if t.track_id == b.track_id)
        gap = abs(tb.bpm * b.ratio / (ta.bpm * a.ratio) - 1)
        if REGISTRY[tr.type].beatmatched:
            assert gap <= 0.08


@pytest.mark.parametrize("name", [n for n, c in REGISTRY.items() if not c.needs_stems])
def test_every_transition_renders(demo_tracks, name):
    a, b = demo_tracks[0], demo_tracks[1]
    t = REGISTRY[name]()
    bars = 8 if t.beatmatched else (2 if name == "reverb_wash" else 0)
    sl = SetList(
        items=[SetItem(a.path, a.track_id, a.title, 4, 28), SetItem(b.path, b.track_id, b.title, 12, 36)],
        transitions=[SetTransition(name, bars)],
    )
    plan = build(sl, {a.track_id: a, b.track_id: b})
    audio = render(plan)
    assert np.isfinite(audio).all()
    assert audio.shape[0] == 2
    expected = TempoMap(plan.tempo, plan.total_beats).time(plan.total_beats)
    assert audio.shape[1] / SR == pytest.approx(expected, abs=9)
    assert np.abs(audio).max() <= 1.0
    assert np.sqrt((audio**2).mean()) > 0.01


def _pair(demo_tracks, vocals_a, vocals_b, shift=1):
    from dataclasses import replace
    a = replace(demo_tracks[0], has_stems=True, bar_vocal=list(vocals_a))
    b = replace(demo_tracks[1], has_stems=True, bar_vocal=list(vocals_b))
    sl = SetList(
        items=[SetItem(a.path, a.track_id, a.title, 4, 52), SetItem(b.path, b.track_id, b.title, 0, 56, 1.0, shift)],
        transitions=[SetTransition("bass_swap", 16)],
    )
    return sl, {a.track_id: a, b.track_id: b}


def test_pitch_shift_glides_back_after_overlap(demo_tracks):
    sl, an = _pair(demo_tracks, [0.0] * 60, [0.0] * 60, shift=1)
    plan = build(sl, an)
    pts = plan.lanes["1:pitch"]
    overlap_end = plan.slots[1].start + 64
    assert np.interp(overlap_end - 1, *zip(*pts)) == pytest.approx(1.0)
    assert pts[-1][1] == pytest.approx(0.0)
    assert pts[-1][0] <= plan.slots[1].end


def test_vocal_handoff_when_both_tracks_sing(demo_tracks):
    # A's singer pauses on bar 40 of the track, which falls inside the overlap
    va = [0.9] * 60
    va[40] = 0.0
    sl, an = _pair(demo_tracks, va, [0.9] * 60, shift=0)
    plan = build(sl, an)
    a_voc, b_voc = plan.lanes["0:stem:vocals"], plan.lanes["1:stem:vocals"]
    w = plan.slots[1].start
    handoff = next(b for b, v in a_voc if v < 0.5)
    assert w < handoff < plan.slots[0].end
    assert np.interp(w + 1, *zip(*b_voc)) == pytest.approx(0.0)
    assert b_voc[-1][1] == pytest.approx(1.0)


def test_lone_singer_stops_on_the_handoff_not_with_the_fader(demo_tracks):
    # by ear: A's voice fading out with A's gain over the second half sounded "противно"; it ends near
    # the middle instead, and B (no vocal) is left alone
    sl, an = _pair(demo_tracks, [0.9] * 60, [0.0] * 60, shift=0)
    plan = build(sl, an)
    w = plan.slots[1].start
    a_voc = plan.lanes["0:stem:vocals"]
    gone = next(x for x, v in a_voc if v < 0.01)
    assert w + 16 <= gone <= w + 48
    assert "1:stem:vocals" not in plan.lanes


def _sung_pair(demo_tracks, name, bars):
    from dataclasses import replace
    sl, an = _pair(demo_tracks, [0.9] * 60, [0.9] * 60, shift=0)
    sl.transitions = [SetTransition(name, bars)]
    for it in sl.items:  # sung lines with a one-beat pause before every bar line
        v8 = [0.0] * (2 * 4 * 60)
        v8[6::8] = [-30.0] * len(v8[6::8])
        v8[7::8] = [-30.0] * len(v8[7::8])
        an[it.track_id] = replace(an[it.track_id], vocal8=v8)
    return sl, an


@pytest.mark.parametrize("name", ["stem_relay", "mashup"])
def test_b_vocal_comes_in_on_a_line_not_a_slow_fade(demo_tracks, name):
    # by ear ("вокал долго вступал", "противно нарастает"): B's voice used to fade in over the last 2-3 bars
    sl, an = _sung_pair(demo_tracks, name, 16)
    plan = build(sl, an)
    end = plan.slots[1].start + 64
    b_voc = plan.lanes["1:stem:vocals"]
    rise = [x for x, v in b_voc if 0.01 < v < 0.99]
    up = next(x for x, v in b_voc if v > 0.99)
    assert up < end - 8  # in before the last two bars
    assert not rise or max(rise) - min(rise) <= 1.0
    a_voc = plan.lanes["0:stem:vocals"]
    assert next(x for x, v in a_voc if v < 0.01) <= up  # never two singers at once


def test_drum_bed_keeps_a_drums_under_b_until_the_end(demo_tracks):
    # v14 21-22 "круто, плавно": A's drums carry under B's intro to the end (covering B's own drum
    # breaks); a v15 half-way kit swap broke it. Basslines still swap half-way (never two basses)
    sl, an = _sung_pair(demo_tracks, "drum_bed", 16)
    plan = build(sl, an)
    mid, end = plan.slots[1].start + 32, plan.slots[1].start + 64
    a_d, a_b, b_b = plan.lanes["0:stem:drums"], plan.lanes["0:stem:bass"], plan.lanes["1:stem:bass"]
    assert np.interp(end - 4, *zip(*a_d)) == pytest.approx(1.0)
    assert np.interp(mid + 1, *zip(*a_b)) == pytest.approx(0.0)
    assert np.interp(mid + 1, *zip(*b_b)) == pytest.approx(1.0)


def test_reverb_wash_is_the_default_for_unblendable_tempo_jumps(demo_tracks):
    from neurojockey.planner.compat import best_ratio
    from neurojockey.planner.planner import _ctx, choose_transition
    a, d = demo_tracks[0], demo_tracks[3]  # 124 -> 150 BPM: not blendable
    _, gap = best_ratio(a.bpm, d.bpm)
    ctx = _ctx(a, d, 0.5, gap, PlanOptions(), [])
    assert not ctx.can_beatmatch
    name, _ = choose_transition(ctx, PlanOptions(), np.random.default_rng(0))
    assert name in {"reverb_wash", "loop_roll"}  # instrumental demo tracks: the roll may ride the tempo


def test_echo_out_is_gone():
    assert "echo_out" not in REGISTRY


def test_quiet_exit_avoids_cutting_a_vocal_line(demo_tracks):
    from dataclasses import replace
    from neurojockey.planner.planner import _quiet_exit
    a = demo_tracks[0]
    beats = [1.0] * (a.n_bars * 4)
    for b in range(4 * 40, 4 * 40 + 3):  # singer pauses around the bar-40 boundary
        beats[b - 1] = 0.0
    a = replace(a, beat_vocal=beats, bar_vocal=[1.0] * a.n_bars)
    assert _quiet_exit(a, 20, 48) == 40


def test_critic_scores_a_planned_set(demo_tracks):
    from neurojockey.critic import critique
    sl = plan_set(demo_tracks, PlanOptions(mode="full", seed=0))
    reps = critique(sl, build(sl, {t.track_id: t for t in demo_tracks}), {t.track_id: t for t in demo_tracks})
    assert len(reps) == 3
    assert all(0 <= r.total <= 1 for r in reps)


def test_program_selects_sections_without_repeating_artists(demo_tracks):
    from dataclasses import replace
    from neurojockey.planner.program import artists, select_and_order
    crate = []
    for i in range(12):
        t = demo_tracks[i % 4]
        crate.append(replace(t, track_id=f"x{i}", title=f"Artist{i % 5} - Song {i}",
                             energy_level=i / 11, bpm=120 + i))
    picked = select_and_order(crate, "club", minutes=8, mode="highlights", hl_bars=16)
    names = [s for _, s in picked]
    assert names[0] == "Warm-up" and "Peak" in names
    per_artist = {}
    for t, _ in picked:
        for a in artists(t):
            per_artist[a] = per_artist.get(a, 0) + 1
    assert max(per_artist.values()) <= 2
    assert len({t.track_id for t, _ in picked}) == len(picked)


@pytest.mark.parametrize("bars", [4, 8, 16])
@pytest.mark.parametrize("name", [n for n, c in REGISTRY.items() if c.beatmatched and c.pre_bars == 0])
def test_transitions_leave_lanes_alone_outside_the_overlap(demo_tracks, name, bars):
    # regression: a transition written for 16 bars, squeezed into 4/8, used to emit backwards ramps
    # that muted B's vocal for the rest of the track
    sl, an = _pair(demo_tracks, [0.9] * 60, [0.9] * 60, shift=0)
    sl.transitions = [SetTransition(name, bars, params={"vocals": "both"})]
    plan = build(sl, an)
    for slot, lo, hi in [(0, plan.slots[0].start, plan.slots[1].start),
                         (1, plan.slots[1].start + 4 * bars, plan.slots[1].end - 16)]:  # minus the set's fade-out
        for lane, pts in plan.lanes_for(slot).items():
            if lane in ("gain", "eq_low", "eq_mid", "eq_high") or lane.startswith("stem:"):
                xs, ys = zip(*pts)
                for beat in np.arange(lo, hi, 1.0):
                    assert np.interp(beat, xs, ys) > 0.5, (lane, beat - plan.slots[slot].start)


def test_handoff_cuts_on_a_breath(demo_tracks):
    from dataclasses import replace
    sl, an = _pair(demo_tracks, [0.9] * 60, [0.9] * 60, shift=0)
    a, b = (an[it.track_id] for it in sl.items)
    breath = 2 * 4 * 42 - 2  # the last beat before bar 42 of A is a pause between lines (inside the overlap)
    v8 = [0.0] * (2 * 4 * 60)
    v8[breath] = v8[breath + 1] = -30.0
    an[a.track_id] = replace(a, vocal8=v8)
    an[b.track_id] = replace(b, vocal8=[0.0] * (2 * 4 * 60))
    plan = build(sl, an)
    a_voc = plan.lanes["0:stem:vocals"]
    cut = next(x for x, v in a_voc if v < 0.01)
    slot = plan.slots[0]
    assert 4 * 42 - 1 <= (slot.src_beat + cut - slot.start) <= 4 * 42  # on the breath, right before bar 42


def test_original_plays_outside_stem_automation(demo_tracks, monkeypatch):
    # a slot with a stem lane must sound exactly like the original wherever the lane is at 1
    from neurojockey.engine import render as R
    from neurojockey.engine.plan import MixPlan

    an = demo_tracks[0]
    full = R._full(an.path)
    rng = np.random.default_rng(0)
    fake = {n: (full * w).astype(np.float32) for n, w in zip(("drums", "bass", "other", "vocals"), rng.dirichlet([1] * 4))}
    monkeypatch.setattr(R.stemlib, "has_stems", lambda tid: True)
    monkeypatch.setattr(R, "_stems", lambda tid: fake)
    slot = Slot(an.path, an.track_id, an.title, bpm=an.bpm * 1.03, t0=an.t0, start=0, beats=128, src_beat=0)
    tm_plan = [(0.0, an.bpm * 1.03), (128.0, an.bpm * 1.03)]
    plain = MixPlan(slots=[slot], tempo=tm_plan)
    stem = MixPlan(slots=[slot], tempo=tm_plan)
    stem.automate(0, "stem:vocals", [(0, 1.0), (64, 1.0), (72, 0.0), (128, 0.0)])
    tm = TempoMap(tm_plan, 128)
    a, _ = R.render_slot(0, slot, plain, tm)
    b, _ = R.render_slot(0, slot, stem, tm)
    cut = int(tm.time(63) * 44100)
    assert np.abs(a[:, 44100:cut] - b[:, 44100:cut]).max() < 1e-6
    late = slice(int(tm.time(90) * 44100), int(tm.time(110) * 44100))
    assert np.abs(a[:, late] - b[:, late]).max() > 1e-3  # vocals really are removed there


def test_window_render_matches_full_render(demo_tracks):
    # partial renders (the critic) must sound like the full slot render inside the window (chunk seams
    # may phase-lock up to CHUNK_ALIGN differently until the chain converges, so compare 50 ms levels)
    from neurojockey.engine import render as R
    from neurojockey.engine.plan import MixPlan

    an = demo_tracks[0]
    slot = Slot(an.path, an.track_id, an.title, bpm=an.bpm, t0=an.t0, start=0, beats=128, src_beat=0)
    tempo = [(0.0, an.bpm * 1.02), (60.0, an.bpm * 1.02), (68.0, an.bpm * 0.98), (128.0, an.bpm * 0.98)]
    plan = MixPlan(slots=[slot], tempo=tempo)
    plan.automate(0, "eq_low", [(56, 1.0), (72, 0.0)])
    tm = TempoMap(tempo, 128)
    full, off = R.render_slot(0, slot, plan, tm)
    t0, t1 = tm.time(56), tm.time(76)
    part, poff = R.render_slot(0, slot, plan, tm, window=(t0, t1))
    a, b = int(t0 * 44100), int(t1 * 44100)
    assert part.shape[1] < full.shape[1] / 2
    x, y = full[:, a - off:b - off], part[:, a - poff:b - poff]
    blk = 2205
    db = lambda z: 10 * np.log10((z[:, :z.shape[1] // blk * blk].reshape(2, -1, blk) ** 2).mean((0, 2)) + 1e-9)
    dd = np.abs(db(x) - db(y))
    assert np.percentile(dd, 95) < 0.3 and dd.max() < 3.0


def _sung(t, amount=0.9):
    from dataclasses import replace
    return replace(t, has_stems=True, bar_vocal=[amount] * t.n_bars)


def test_sung_tracks_get_no_pitch_shift(demo_tracks):
    # a singer is never pitch-shifted (no warped-tape key glide)
    a, b = _sung(demo_tracks[0]), _sung(demo_tracks[1])  # 8A -> 9A: a +-1 st shift would help the key
    sl = plan_set([a, b], PlanOptions(mode="highlights", keep_order=True, hl_bars=16, max_shift=1))
    assert all(it.pitch_shift == 0 for it in sl.items)


def test_cut_stops_the_vocal_on_a_breath(demo_tracks):
    from dataclasses import replace
    va = [0.9] * 60
    sl, an = _pair(demo_tracks, va, [0.0] * 60, shift=0)
    sl.transitions = [SetTransition("cut", 0, params={"entry": "drop"})]
    a = an[sl.items[0].track_id]
    v8 = [0.0] * (8 * 60)
    end8 = 8 * sl.items[0].end_bar
    v8[end8 - 2] = -30.0  # a breath on the last beat before the cut, singing across the cut otherwise
    an[a.track_id] = replace(a, vocal8=v8)
    plan = build(sl, an)
    slot = plan.slots[0]
    pts = plan.lanes["0:stem:vocals"]
    assert np.interp(slot.end - 0.2, *zip(*pts)) < 0.01  # silent by the cut
    assert np.interp(slot.end - 2.0, *zip(*pts)) > 0.99  # still singing before the breath


def test_effect_moves_stay_rare(demo_tracks):
    from neurojockey.planner.compat import best_ratio
    from neurojockey.planner.planner import _ctx, choose_transition
    a, b = demo_tracks[0], demo_tracks[1]
    _, gap = best_ratio(a.bpm, b.bpm)
    ctx = _ctx(a, b, 0.2, gap, PlanOptions(), ["riser", "drum_bed", "filter"])
    name, _ = choose_transition(ctx, PlanOptions(), np.random.default_rng(0))
    assert name not in {"gate", "loop_roll", "riser", "cut", "reverb_wash"}


def test_rap_without_pauses_cuts_on_a_downbeat_into_a_vocal_reverb(demo_tracks):
    # by ear: a stop on a quick breath inside a verse still sounds cut ("выход обрезан"); with no real pause
    # A's vocal stops on a bar line and its last syllable rings out in a vocal-only reverb (picked by ear)
    from dataclasses import replace
    sl, an = _pair(demo_tracks, [0.9] * 60, [0.9] * 60, shift=0)
    for it in sl.items:
        v8 = [0.0] * (2 * 4 * 60)
        v8[3::4] = [-30.0] * len(v8[3::4])  # a quick breath every two beats, never a beat of silence
        an[it.track_id] = replace(an[it.track_id], vocal8=v8)
    plan = build(sl, an)
    w = plan.slots[1].start
    cut = next(x for x, v in plan.lanes["0:stem:vocals"] if v < 0.99 and x > w)
    assert (cut - w) % 4 < 0.3  # the fade starts on the downbeat
    send = plan.lanes["0:send_vreverb"]
    assert np.interp(cut - 0.1, *zip(*send)) == pytest.approx(1.0)
    assert np.interp(cut + 4, *zip(*send)) == pytest.approx(0.0)


def test_double_drop_hands_the_vocal_over_on_a_line(demo_tracks):
    # every layering move goes through the line-aware handoff; double_drop used to fade A's voice at W
    sl, an = _sung_pair(demo_tracks, "double_drop", 16)
    plan = build(sl, an)
    w = plan.slots[1].start
    gone = next(x for x, v in plan.lanes["0:stem:vocals"] if v < 0.01)
    up = next(x for x, v in plan.lanes["1:stem:vocals"] if v > 0.99)
    assert w - 4 <= gone <= w + 5
    assert up >= w + 32 and up >= gone


def test_half_time_chain_keeps_overlaps_at_their_own_length(demo_tracks):
    # a run of tracks at ratio 0.5 sits on a half-speed master grid: 16 own bars must not become 16 grid bars
    from dataclasses import replace
    from neurojockey.planner.planner import plan_set
    fast = [replace(t, track_id=f"h{i}", bpm=130.0 + i, variable_tempo=False) for i, t in enumerate(demo_tracks)]
    slow = replace(demo_tracks[0], track_id="slow", bpm=66.0, variable_tempo=False)
    sl = plan_set([slow] + fast, PlanOptions(mode="full", keep_order=True, forced=["bass_swap"]))
    halved = [tr for i, tr in enumerate(sl.transitions)
              if sl.items[i].ratio == 0.5 and sl.items[i + 1].ratio == 0.5]
    assert halved and all(tr.bars <= 8 for tr in halved)


def test_no_pause_exit_picks_slowed_when_the_set_does_not_lift(demo_tracks):
    # reverb where the energy goes up, a soft slowed + reverb where B is calmer and leaves room for the tail
    from dataclasses import replace
    for b_energy, lane in ((0.2, "0:send_vslow"), (0.9, "0:send_vreverb")):
        sl, an = _pair(demo_tracks, [0.9] * 60, [0.0] * 60, shift=0)
        a_id, b_id = (it.track_id for it in sl.items)
        v8 = [0.0] * (2 * 4 * 60)
        v8[3::4] = [-30.0] * len(v8[3::4])  # raps straight through: never a beat of silence
        an[a_id] = replace(an[a_id], vocal8=v8, energy_level=0.5)
        an[b_id] = replace(an[b_id], vocal8=[-60.0] * (2 * 4 * 60), energy_level=b_energy)
        plan = build(sl, an)
        assert lane in plan.lanes, (b_energy, [k for k in plan.lanes if "send" in k])


def test_b_raps_straight_through_enters_on_a_downbeat_out_of_a_swell(demo_tracks):
    # by ear ("резкое вступление вокала"): no mid-verse entry after a quick breath; without a real phrase
    # start B comes in on a downbeat, its first word growing out of a reverse-reverb swell
    from dataclasses import replace
    sl, an = _pair(demo_tracks, [0.9] * 60, [0.9] * 60, shift=0)
    v8 = [0.0] * (2 * 4 * 60)
    v8[3::4] = [-30.0] * len(v8[3::4])  # a quick breath every two beats, never a beat of silence
    for it in sl.items:
        an[it.track_id] = replace(an[it.track_id], vocal8=v8)
    plan = build(sl, an)
    w = plan.slots[1].start
    up = next(x for x, v in plan.lanes["1:stem:vocals"] if v > 0.99)
    assert (up - w) % 4 == pytest.approx(0)
    swell = plan.lanes["1:send_vswell"]
    assert np.interp(up + 0.5, *zip(*swell)) == pytest.approx(1.0)
