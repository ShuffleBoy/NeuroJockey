"""Plan-level judge for tuning iterations, no rendering. Usage: uv run python tools/judge.py [club|all|both]"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from neurojockey.analysis import analyze
from neurojockey.audio import find_tracks
from neurojockey.critic import critique
from neurojockey.engine.tempo import TempoMap
from neurojockey.planner import PlanOptions, build, plan_set
from neurojockey.planner.mood import attach_moods
from neurojockey.planner.program import select_and_order

GIMMICK = {"gate", "loop_roll", "riser", "cut", "reverb_wash"}


def lane(plan, k, name, x, default=1.0):
    pts = plan.lanes.get(f"{k}:{name}")
    return float(np.interp(x, [p[0] for p in pts], [p[1] for p in pts])) if pts else default


def track_bar(an, slot, beat):
    return (slot.src_beat + beat - slot.start) / (4 * slot.bpm / an.bpm)


def judge(sl, analyses, label):
    plan = build(sl, analyses)
    tm = TempoMap(plan.tempo, plan.total_beats)
    m = Counter()
    types = Counter(t.type for t in sl.transitions)
    reps = critique(sl, plan, analyses)
    for i, tr in enumerate(sl.transitions):
        a, b = plan.slots[i], plan.slots[i + 1]
        aa, ab = analyses[a.track_id], analyses[b.track_id]
        w, e = b.start, a.end
        for x in np.arange(w, e, 0.5):
            dt = tm.time(x + 0.5) - tm.time(x)
            ga, gb = lane(plan, i, "gain", x), lane(plan, i + 1, "gain", x)
            ba, bb = track_bar(aa, a, x), track_bar(ab, b, x)
            va = aa.vocal_on(int(ba * 4)) * lane(plan, i, "stem:vocals", x) * ga
            vb = ab.vocal_on(int(bb * 4)) * lane(plan, i + 1, "stem:vocals", x) * gb
            if va > 0.4 and vb > 0.4:
                m["vocal_layer"] += dt
            if ga > 0.5 and gb > 0.5:
                m["music_layer"] += dt
        r = reps[i]
        if r.scores.get("key", 1) < 0.5 and e > w:
            m["key_bad"] += tm.time(e) - tm.time(w)
        if r.scores.get("vocal_cut", 1) < 0.6:
            m["midline"] += 1
        it_a, it_b = sl.items[i], sl.items[i + 1]
        seg = ab.segment_at(int(it_b.start_bar))
        if seg and seg.label != "intro" and it_b.start_bar - seg.start_bar >= 4:
            m["entry_mid"] += 1
        L = max(tr.bars, 0) / it_a.ratio
        if any(s.label == "drop" and it_a.end_bar - L + 0.5 < s.start_bar < it_a.end_bar - 0.5 for s in aa.segments):
            m["a_drop"] += 1
        if L == 0:
            ends = [s.end_bar for s in aa.segments]
            if min(abs(it_a.end_bar - x) for x in ends) > 2:
                m["exit_mid"] += 1
            ea = aa.bar_energy[min(int(it_a.end_bar) - 1, aa.n_bars - 1)]
            eb = ab.bar_energy[min(int(it_b.start_bar), ab.n_bars - 1)]
            if eb < 0.6 * ea:
                m["hole"] += 1
        if b.pitch_shift and ab.vocal_amount(int(it_b.start_bar), int(it_b.start_bar) + int(L) + 8) > 0.3:
            m["pitch_voc"] += 1
        if it_a.end_bar > aa.active_end + 2:
            m["fade_exit"] += 1  # A still playing into its fade-out / drum-less tail
        if tr.type in GIMMICK:
            m["gimmick"] += 1
    en = [analyses[it.track_id].energy_level for it in sl.items]
    m["energy_jump"] = sum(abs(y - x) > 0.2 for x, y in zip(en, en[1:]))  # > 2 levels of 10
    n = len(sl.transitions)
    score = np.mean([r.total for r in reps])
    print(f"== {label}: {n} transitions, critic {score:.3f}")
    print("   " + "  ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(m.items())))
    print("   types: " + ", ".join(f"{k} {v}" for k, v in types.most_common()))
    return m, plan


def club():
    crate = [analyze(p, with_stems=False) for p in find_tracks([Path("tracks")])]
    attach_moods(crate, use_llm=True)
    picked = select_and_order(crate, "club", 60, "highlights", 48, 1)
    ans = [analyze(t.path, with_stems=True) for t, _ in picked]
    opts = PlanOptions(mode="highlights", keep_order=True, duration_min=60, program="club",
                       sections=[s for _, s in picked], hl_bars=48, max_shift=1)
    return plan_set(ans, opts), {a.track_id: a for a in ans}


def all_tracks():
    ans = [analyze(p, with_stems=True) for p in find_tracks([Path("tracks")])]
    opts = PlanOptions(mode="highlights", hl_bars=48, max_shift=1)
    return plan_set(ans, opts), {a.track_id: a for a in ans}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("club", "both"):
        judge(*club(), "club")
    if which in ("all", "both"):
        judge(*all_tracks(), "all")
