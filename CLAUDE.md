# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NeuroJockey (package `neurojockey`, CLI `nj`) is an offline AI DJ (Python 3.12, uv). It takes a folder of tracks and
produces a beatmatched mix: it analyses each track, splits stems with Demucs, plans the set and renders it. Mix quality
is judged by ear; critic and judge scores are only proxies.

## Commands

```bash
uv sync
uv run pytest -q                                   # ~30 s; synthetic tracks, no real audio needed
uv run pytest tests/test_engine.py -k handoff      # single test
uv run python tools/judge.py all|club|both         # plan-level metrics over ./tracks, no rendering

uv run nj mix tracks -o out/mix.mp3                          # preset highlights (48 bars per track)
uv run nj mix tracks -o out/club.mp3 -p club -d 60           # presets: full | highlights | club | warmup | peak
uv run nj render out/X.set.json                    # re-render a (hand-edited) SetList
```

- Analysis and stems are cached per track in `~/.cache/neurojockey/<content-hash>/` (`NEUROJOCKEY_CACHE`). A new track
  costs about 25 s; full renders of big folders take tens of minutes, so run them in the background.
- The CLI is deliberately minimal (presets in `cli.py:PRESETS`); other knobs live in `PlanOptions` and
  `render(draft=...)`.
- `find_tracks` recurses into subfolders and skips groups of byte-identical files saved under different song names (
  broken downloads).

## Pipeline (spans many files)

`analysis/` → `stems.py` → `planner/` → **SetList** (editable JSON: items with regions, ratio, pitch shift, and
transitions with type and bars) → `planner/builder.py` → **MixPlan** (`engine/plan.py`) → `engine/render.py` → critic.

- **Analysis** (`analysis/analyze.py`, model in `analysis/model.py`):
    - beat grid from beat_this plus a constant-tempo fit (`variable_tempo` flag), key from stem chroma (Camelot),
      sections and highlights;
    - stem features: `bar_stem_db`, `vocal8` (vocal level per 8th note, used for breath detection), `bar_vocal`.
- **Planner** (`planner/planner.py`):
    - order (`order.py`, or `program.py` for the club/warmup/peak presets);
    - tempo ratio including half/double time (`compat.best_ratio`), harmonic shift, play regions (highlights mode);
    - transition choice: `choose_transition` scores every class in `transitions/library.py` against a `Ctx` (
      `transitions/base.py`: key, bpm_gap, groove_change, stems, vocal tail/head, room, recent).
    - `can_beatmatch` = gap ≤ 17% and no variable tempo. Otherwise the choice falls back to non-beatmatched moves (
      reverb_wash, loop_roll).
- **Transitions** are classes with `bars()`, `score()` and `apply(win)`. `apply` writes automation lanes on a `Window`:
  `gain`, `eq_*`, `filter`, `send_reverb`, `stem:<name>`, plus source edits and master FX. Comments in `score()` record
  the listening verdict behind each rule; keep that convention.
- **Builder** (`planner/builder.py`) turns the SetList into slots on a global beat grid with a piecewise tempo curve. It
  adds the automatic mechanics:
    - meet-in-the-middle tempo ride, harmonic pitch return, set ending;
    - **vocal handoff** (`_vocal_handoffs`): every layering transition names a window via `Transition.handoff()`. A
      stops at a line end there, preferring a real pause over an in-verse breath, and B enters at the start of its next
      line, searched in track 8ths via `_eighth()`. It runs whenever either deck sings and replaces the transition's own
      vocal lanes. Without a pause, `vocal_exit` picks how A's voice leaves.
    - Lane beats are **global** beats. Converting to a track's own beats needs `ratio = slot.bpm / an.bpm`; getting this
      wrong in half-time cuts vocals off.
- **Renderer** (`engine/render.py`):
    - stem lanes are mixed into the source *before* time-stretching (`src = full + Σ stem·(env−1)`);
    - bar-sized Rubber Band chunks pinned to the master grid;
    - `window=` renders only part of a slot (the critic's kick-sync check), with 16 beats of pre-roll so the chunk phase
      chain converges.
- **Critic** (`critic.py`): a per-transition QA report printed at the end of `mix`/`render` and written to the `.md`:
  phrase alignment, vocal cut, kick sync on rendered overlaps, loudness holes and bumps. `SET SCORE` / `min`.

Mixing rules the code enforces: never cut a vocal mid-word, never layer two lead vocals or two basslines, swap drum kits
together with the bass, keep FX moves rare (`_fx_spacing`).

## Conventions

- LLM features (`llm.py`) call any **OpenAI-compatible** chat completions API configured in `.env`: `LLM_BASE_URL`,
  `LLM_API_KEY`, `LLM_MODEL` (see `.env.example`). Don't add provider SDKs. `.env` is gitignored.
- `out/` and `tracks/` stay out of git.
