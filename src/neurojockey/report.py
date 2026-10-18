"""Human-readable tracklist / cue sheet for a rendered mix."""

from .analysis import TrackAnalysis
from .analysis.key import shift_camelot
from .engine.plan import MixPlan, SetList
from .engine.tempo import TempoMap
from .transitions import get


def _ts(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


def tracklist(sl: SetList, plan: MixPlan, analyses: dict[str, TrackAnalysis]) -> str:
    tm = TempoMap(plan.tempo, plan.total_beats)
    total = tm.time(plan.total_beats)
    lines = [f"# AI DJ set — {len(sl.items)} tracks, {_ts(total)}", ""]
    if sl.notes:
        lines += [f"> {sl.notes}", ""]
    sectioned = any(it.section for it in sl.items)
    lines += ["| # | Cue | Track | BPM | Key | Energy | Plays |", "|---|---|---|---|---|---|---|"]
    current = None
    for i, (it, slot) in enumerate(zip(sl.items, plan.slots)):
        if sectioned and it.section != current:
            current = it.section
            lines.append(f"| | **{_ts(tm.time(slot.start))}** | **— {it.section} —** | | | | |")
        an = analyses[it.track_id]
        bpm = f"{an.bpm:g}" + (f" (x{it.ratio:g})" if it.ratio != 1 else "")
        key = an.camelot + (f" ({shift_camelot(an.camelot, it.pitch_shift)} {it.pitch_shift:+d} st while mixing in)" if it.pitch_shift else "")
        bar = "▮" * max(1, round(an.energy_level * 10))
        dur = tm.time(slot.end) - tm.time(slot.start)
        lines.append(f"| {i + 1} | {_ts(tm.time(slot.start))} | {it.title} | {bpm} | {key} | {bar} | "
                     f"bars {it.start_bar:g}-{it.end_bar:g} ({_ts(dur)}) |")
    lines += ["", "## Transitions", ""]
    for i, tr in enumerate(sl.transitions):
        t = get(tr.type)
        at = tm.time(plan.slots[i + 1].start)
        span = f"{tr.bars:g}-bar overlap" if tr.bars > 0 else ("short pause" if tr.bars < 0 else "on the phrase")
        lines.append(f"**{_ts(at)} · {t.title}** ({span}) — {sl.items[i].title} → {sl.items[i + 1].title}  ")
        lines.append(f"{t.description}  ")
        if tr.reason:
            lines.append(f"_Why:_ {tr.reason}")
        lines.append("")
    if plan.events:
        lines += ["## Stem / key moves", ""]
        for ev in sorted(plan.events, key=lambda e: e["beat"]):
            lines.append(f"- **{_ts(tm.time(ev['beat']))}** {ev['text']}")
    return "\n".join(lines) + "\n"
