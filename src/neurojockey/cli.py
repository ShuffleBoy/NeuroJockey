"""NeuroJockey command line."""

import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

warnings.filterwarnings("ignore")

app = typer.Typer(add_completion=False, no_args_is_help=True, help="NeuroJockey, an offline AI DJ: point it at a folder of tracks and get a mix.")
console = Console()


def _analyze_all(paths: list[Path], stems: bool):
    from .analysis import analyze
    from .audio import find_tracks

    files = find_tracks(paths, warn=lambda m: console.print(f"[yellow]{m}[/]"))
    if not files:
        console.print("[red]no audio files found[/]")
        raise typer.Exit(1)
    out = []
    for f in track(files, description="analyzing", console=console):
        out.append(analyze(f, with_stems=stems))
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


def _render_set(sl, analyses, out: Path):
    from .audio import save
    from .engine.render import render
    from .planner import build
    from .report import tracklist

    from .critic import critique, summary

    plan = build(sl, analyses)
    probes = {}
    with console.status("rendering"):
        audio = render(plan, probes=probes)
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
    console.print(f"[green]set:[/] {out.with_suffix('.set.json')}  (edit and run `nj render` to re-render)")


PRESETS = {
    "full": dict(mode="full"),  # every track, start to finish
    "highlights": dict(mode="highlights"),  # the best part of every track
    "club": dict(mode="highlights", program="club"),  # warm-up → dance → breather → peak → closing
    "warmup": dict(mode="highlights", program="warmup"),  # opening → groove
    "peak": dict(mode="highlights", program="peak"),  # peak → release
}
HL_BARS = 48


@app.command()
def mix(paths: list[Path],
        out: Path = typer.Option(Path("out/mix.mp3"), "-o", help="output .wav/.mp3/.flac"),
        preset: str = typer.Option("highlights", "-p", "--preset", help=" | ".join(PRESETS)),
        duration: float | None = typer.Option(None, "-d", "--duration", help="target length in minutes")):
    """Analyze, plan and render a DJ mix."""
    from .planner import PlanOptions, plan_set

    if preset not in PRESETS:
        raise typer.BadParameter(f"unknown preset {preset!r}, choose from: {', '.join(PRESETS)}")
    mode, program = PRESETS[preset]["mode"], PRESETS[preset].get("program")
    if program:
        # phase 1: fast analysis of the whole crate (no stems) and track selection
        from .planner.mood import attach_moods
        from .planner.program import select_and_order

        crate = _analyze_all(paths, stems=False)
        with console.status("rating track moods"):
            attach_moods(crate, log=console.print)
        picked = select_and_order(crate, program, duration or 60, mode, HL_BARS, 1, log=console.print)
        console.print(f"[bold]selected {len(picked)} of {len(crate)} tracks[/]")
        # phase 2: stems only for the selected tracks
        from .analysis import analyze

        ans = [analyze(t.path, with_stems=True) for t, _ in track(picked, description="stems", console=console)]
        sections = [s for _, s in picked]
    else:
        ans = _analyze_all(paths, stems=True)
        sections = None
    console.print(_analysis_table(ans))
    opts = PlanOptions(mode=mode, keep_order=bool(program), duration_min=duration, program=program,
                       sections=sections, hl_bars=HL_BARS, log=console.print)
    _render_set(plan_set(ans, opts), {a.track_id: a for a in ans}, out)


@app.command()
def render(set_file: Path, out: Path | None = typer.Option(None, "-o")):
    """Re-render an (edited) .set.json."""
    from .analysis import analyze
    from .engine.plan import SetList

    sl = SetList.load(set_file)
    analyses = {}
    for it in sl.items:
        an = analyze(it.path, with_stems=True)
        analyses[an.track_id] = an
        it.track_id = an.track_id
    _render_set(sl, analyses, out or set_file.with_name(set_file.name.replace(".set.json", "") + ".mp3"))


if __name__ == "__main__":
    app()
