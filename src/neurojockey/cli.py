"""NeuroJockey command line."""

import json
import warnings
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

warnings.filterwarnings("ignore")

app = typer.Typer(add_completion=False, no_args_is_help=True, help="NeuroJockey, an offline AI DJ: analyze tracks, split stems, plan and render mixes.")
console = Console()


def _analyze_all(paths: list[Path], stems: bool, force: bool = False):
    from .analysis import analyze
    from .audio import find_tracks

    files = find_tracks(paths, warn=lambda m: console.print(f"[yellow]{m}[/]"))
    if not files:
        console.print("[red]no audio files found[/]")
        raise typer.Exit(1)
    out = []
    for f in track(files, description="analyzing", console=console):
        out.append(analyze(f, with_stems=stems, force=force))
    return out


def _analysis_table(ans) -> Table:
    t = Table(title=f"Tracks ({len(ans)})")
    for col in ("Track", "BPM", "Key", "Camelot", "LUFS", "Energy", "Bars", "Structure", "Best highlight"):
        t.add_column(col)
    for a in ans:
        struct = " ".join(f"{s.label[:2]}{s.bars}" for s in a.segments)
        h = a.highlights[0]
        bpm = f"{a.bpm:g}" + (" ~" if a.variable_tempo else "")
        t.add_row(a.title, bpm, a.key, a.camelot, f"{a.lufs:.1f}", f"{a.energy_level:.2f}", str(a.n_bars),
                  struct, f"{h.kind} {h.start_bar}-{h.end_bar}")
    return t


@app.command()
def analyze(paths: list[Path], no_stems: bool = typer.Option(False, "--no-stems", help="skip source separation"),
            force: bool = typer.Option(False, help="ignore cache"),
            json_out: Path | None = typer.Option(None, "--json", help="dump analyses to a JSON file")):
    """Analyze tracks: BPM/beat grid, key, loudness, structure, highlights (+ stems)."""
    ans = _analyze_all(paths, not no_stems, force)
    console.print(_analysis_table(ans))
    if json_out:
        json_out.write_text(json.dumps([a.to_dict() for a in ans], indent=1))


@app.command()
def stems(paths: list[Path], out: Path = typer.Option(Path("out/stems"), "-o", help="output folder")):
    """Split tracks into vocals / drums / bass / other and export them as WAV."""
    from . import stems as stemlib
    from .audio import find_tracks, save
    from .cache import track_id

    for f in find_tracks(paths):
        tid = track_id(f)
        console.print(f"[bold]{f.name}[/]")
        stemlib.separate(f, tid)
        for name, audio in stemlib.load_stems(tid).items():
            dest = out / f.stem / f"{name}.wav"
            save(dest, audio)
            console.print(f"  {dest}")


@app.command()
def transitions():
    """List available transitions."""
    from .transitions import REGISTRY

    t = Table(title="Transitions")
    for col in ("name", "title", "stems", "beatmatched", "description"):
        t.add_column(col)
    for name, cls in REGISTRY.items():
        t.add_row(name, cls.title, "yes" if cls.needs_stems else "", "yes" if cls.beatmatched else "", cls.description)
    console.print(t)


def _render_set(sl, analyses, out: Path, draft: bool = False):
    from .audio import save
    from .engine.render import render
    from .planner import build
    from .report import tracklist

    from .critic import critique, summary

    plan = build(sl, analyses)
    probes = {}
    with console.status("rendering"):
        audio = render(plan, draft=draft, probes=probes)
    save(out, audio)
    sl.save(out.with_suffix(".set.json"))
    report = tracklist(sl, plan, analyses)
    if sl.transitions:
        with console.status("checking the mix"):
            qa = summary(critique(sl, plan, analyses, audio, probes=probes))
        report += "\n## Mix QA\n\n```\n" + qa + "\n```\n"
    out.with_suffix(".md").write_text(report)
    console.print(report)
    console.print(f"[green]mix:[/] {out}  ({audio.shape[1] / 44100 / 60:.1f} min)")
    console.print(f"[green]set:[/] {out.with_suffix('.set.json')}  (edit and run `neurojockey render` to re-render)")


@app.command()
def mix(paths: list[Path],
        out: Path = typer.Option(Path("out/mix.wav"), "-o", help="output .wav/.mp3/.flac"),
        mode: str = typer.Option("full", help="full = whole tracks, highlights = only the best parts"),
        duration: float | None = typer.Option(None, help="target length in minutes"),
        keep_order: bool = typer.Option(False, help="play tracks in the given order"),
        allow: str | None = typer.Option(None, help="comma-separated transitions to choose from"),
        force_transitions: str | None = typer.Option(None, "--transitions", help="comma-separated transitions to use in order; 'auto' = let the planner pick that one"),
        curve: str = typer.Option("peak", help="energy curve: peak | rise | flat"),
        hl_bars: int = typer.Option(32, help="bars per highlight in highlights mode"),
        max_shift: int = typer.Option(1, help="max semitones of harmonic pitch shift (0 = off)"),
        no_stems: bool = typer.Option(False, "--no-stems", help="skip stems (no stem transitions)"),
        program: str | None = typer.Option(None, help="pick tracks from the folder along a set plan: club | warmup | peak"),
        director: str = typer.Option("rules", help="rules | llm (OpenAI-compatible API, see .env.example)"),
        llm_model: str | None = typer.Option(None, help="model id (default: LLM_MODEL)"),
        llm_mood: bool = typer.Option(True, help="with --program: ask the LLM which tracks are sad / club bangers"),
        draft: bool = typer.Option(False, help="fast, lower-quality time-stretch (for auditioning)"),
        seed: int = typer.Option(0)):
    """Analyze, plan and render a DJ mix."""
    from .planner import PlanOptions, plan_set

    if program:
        # phase 1: fast analysis of the whole crate (no stems) and track selection
        from .planner.program import select_and_order

        from .planner.mood import attach_moods

        crate = _analyze_all(paths, stems=False)
        with console.status("rating track moods"):
            attach_moods(crate, use_llm=llm_mood, model=llm_model, log=console.print)
        picked = select_and_order(crate, program, duration or 60, mode, hl_bars, max_shift, log=console.print)
        console.print(f"[bold]selected {len(picked)} of {len(crate)} tracks[/]")
        # phase 2: stems only for the selected tracks
        from .analysis import analyze as _an

        ans = [_an(t.path, with_stems=not no_stems) for t, _ in track(picked, description="stems", console=console)]
        sections = [s for _, s in picked]
    else:
        ans = _analyze_all(paths, not no_stems)
        sections = None
    console.print(_analysis_table(ans))
    analyses = {a.track_id: a for a in ans}
    opts = PlanOptions(mode=mode, keep_order=keep_order or bool(program), duration_min=duration,
                       program=program, sections=sections,
                       transitions=allow.split(",") if allow else None,
                       forced=force_transitions.split(",") if force_transitions else None,
                       hl_bars=hl_bars, seed=seed, max_shift=max_shift, energy_curve=curve,
                       use_stems=not no_stems, log=console.print)
    if director == "llm":
        from .planner.director import llm_plan

        with console.status("asking the LLM director to plan the set"):
            sl = llm_plan(ans, opts, llm_model)
    else:
        sl = plan_set(ans, opts)
    _render_set(sl, analyses, out, draft)


@app.command()
def render(set_file: Path, out: Path | None = typer.Option(None, "-o"), draft: bool = typer.Option(False, help="fast, lower-quality time-stretch (for auditioning)")):
    """Re-render an (edited) .set.json."""
    from .analysis import analyze
    from .engine.plan import SetList

    sl = SetList.load(set_file)
    analyses = {}
    for it in sl.items:
        an = analyze(it.path, with_stems=True)
        analyses[an.track_id] = an
        it.track_id = an.track_id
    _render_set(sl, analyses, out or set_file.with_name(set_file.name.replace(".set.json", "") + ".wav"), draft)


@app.command()
def preview(a: Path, b: Path, transition: str = typer.Argument(..., help="transition name"),
            out: Path | None = typer.Option(None, "-o"), lead: int = typer.Option(8, help="bars before/after"),
            draft: bool = typer.Option(False, help="fast, lower-quality time-stretch (for auditioning)")):
    """Render a short clip of one transition between two tracks (for auditioning)."""
    from .analysis import analyze
    from .engine.plan import SetItem, SetList, SetTransition
    from .planner.compat import best_ratio, best_shift
    from .planner.planner import PlanOptions, _ctx
    from .transitions import get

    ta, tb = analyze(a), analyze(b)
    r, gap = best_ratio(ta.bpm, tb.bpm)
    shift, key = best_shift(ta.camelot, tb.camelot)
    t = get(transition)
    ctx = _ctx(ta, tb, key, gap, PlanOptions(mode="highlights"), [])
    bars = t.bars(ctx)
    ha = ta.highlights[0]
    a_end = min(ta.n_bars, ha.end_bar + bars)
    a_start = max(0, a_end - bars - max(lead, t.pre_bars + 4))
    entry = t.entry(ctx)
    hb = tb.highlights[0].start_bar
    b_start = hb if entry == "drop" else max(0, hb - int(np.ceil(bars / r)))
    b_end = min(tb.n_bars, b_start + int(np.ceil(bars / r)) + lead)
    sl = SetList(items=[SetItem(ta.path, ta.track_id, ta.title, a_start, a_end),
                        SetItem(tb.path, tb.track_id, tb.title, b_start, b_end, r, shift)],
                 transitions=[SetTransition(transition, bars, {"entry": entry})])
    _render_set(sl, {ta.track_id: ta, tb.track_id: tb}, out or Path(f"out/preview_{transition}.wav"), draft)


@app.command()
def demo(out: Path = typer.Option(Path("tracks"), "-o"), n: int = typer.Option(5)):
    """Generate synthetic demo tracks with known BPM / key / structure."""
    from .audio import save
    from .synth import generate

    specs = [(124, 9, "minor"), (126, 4, "minor"), (122, 0, "major"), (128, 2, "minor"), (174, 7, "minor"),
             (120, 5, "minor"), (125, 11, "minor"), (130, 7, "major")]
    for i, (bpm, pc, mode) in enumerate(specs[:n]):
        dest = out / f"demo{i + 1}_{bpm}bpm.wav"
        save(dest, generate(bpm, pc, mode, seed=i))
        console.print(f"{dest}")




if __name__ == "__main__":
    app()
