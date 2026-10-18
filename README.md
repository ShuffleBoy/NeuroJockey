# NeuroJockey

[![tests](https://github.com/ShuffleBoy/NeuroJockey/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/ShuffleBoy/NeuroJockey/actions/workflows/tests.yml)
[![version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FShuffleBoy%2FNeuroJockey%2Fmaster%2Fpyproject.toml&query=%24.project.version&label=version)](pyproject.toml)
[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

NeuroJockey turns a folder of tracks into a DJ mix. It finds the beats, splits each track into vocals, drums,
bass and the rest, picks an order and a transition for every pair, and renders one beatmatched file. It all
runs on your own machine.

Most of the code was written by AI. A human listened to the results and complained ("mush", "too abrupt",
"vocal cut off", "the drums don't line up") until there was nothing left to complain about. We think it sounds
good, but we're biased.

## Getting started

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and `ffmpeg`. ffmpeg is only used to read `.m4a`
files.

```bash
git clone https://github.com/ShuffleBoy/NeuroJockey && cd NeuroJockey
uv sync
```

The first `uv sync` is large. It installs PyTorch, Demucs, beat_this, librosa and pedalboard (which bundles
Rubber Band), and the model weights download the first time you run it. On Apple Silicon the models run on
MPS. Anywhere else they run on the CPU, which works but takes a while.

In the examples below, `tracks` is any folder of music. Subfolders are included.

```bash
uv run nj demo                                         # synthetic tracks in tracks/, if you have none
uv run nj mix tracks -o out/mix.mp3                    # every track, start to finish
uv run nj mix tracks -o out/hl.mp3 --mode highlights --duration 30                    # best parts, 30 minutes
uv run nj mix tracks -o out/club.mp3 --program club --duration 60 --mode highlights   # a one-hour club set
uv run nj mix tracks -o out/llm.mp3 --director llm     # let the LLM director plan the set (see below)
uv run nj render out/hl.set.json                       # re-render after editing the set by hand
```

`nj` is short for `neurojockey`. Both names run the same command.

| command | what it does |
|---|---|
| `analyze PATHS` | shows BPM, beat grid, key, loudness, structure and best moments |
| `stems PATHS -o DIR` | splits tracks into vocals, drums, bass and other |
| `transitions` | lists the available transitions |
| `mix PATHS` | analyses, plans and renders a set, and writes a `.set.json` and a `.md` report next to it |
| `render SET.json` | renders a set file you edited by hand |
| `preview A B NAME` | renders a single transition between two tracks |
| `demo` | generates synthetic tracks to try things on |

### `mix` options

What to play:

| option | default | what it does |
|---|---|---|
| `-o FILE` | `out/mix.wav` | output file: `.wav`, `.mp3` or `.flac` |
| `--mode full\|highlights` | `full` | whole tracks, or only the best part of each |
| `--hl-bars N` | `32` | how many bars of each track to play in highlights mode |
| `--duration MIN` | | target length of the set in minutes |
| `--program club\|warmup\|peak` | | pick tracks from the folder and arrange them as a set (see below) |
| `--keep-order` | off | play the tracks in the order you gave them |
| `--curve peak\|rise\|flat` | `peak` | energy curve of the set when it picks the order |

How to mix:

| option | default | what it does |
|---|---|---|
| `--allow a,b,...` | all | transitions it may choose from, e.g. `bass_swap,filter` |
| `--transitions a,b,...` | | use these transitions in this order; `auto` lets it pick one slot |
| `--max-shift N` | `1` | how many semitones it may shift a key; `0` turns it off |
| `--no-stems` | off | skip stem separation (and the transitions that need stems) |
| `--seed N` | `0` | changes how near-ties are broken, for a slightly different set |
| `--draft` | off | about five times faster time-stretching, at lower quality |

LLM:

| option | default | what it does |
|---|---|---|
| `--director rules\|llm` | `rules` | who plans the set: the built-in rules or a language model |
| `--llm-model ID` | `LLM_MODEL` | override the model from `.env` |
| `--no-llm-mood` | | with `--program`, guess moods from the audio without asking the LLM |

`--program` picks the tracks itself and splits the set into sections, each with its own energy target:

- `club`: warm-up → dance → breather → peak → closing
- `warmup`: opening → groove, for the start of a night
- `peak`: peak → release, for peak time

Each track is analysed once, which takes about 25 seconds. The analysis and stems are cached in
`~/.cache/neurojockey/` (set `NEUROJOCKEY_CACHE` to move it), so the next run loads the track straight away.

## LLM director

`--director llm` gives the track analyses and the list of transitions to a language model, and the model
chooses the order and the transitions. The planner still checks every choice, so a bad suggestion can't
break the mix. With `--program`, the model also says which tracks belong on the dance floor and which are
sad songs about an ex.

Any OpenAI-compatible API works, including OpenAI, OpenRouter and local servers. Copy `.env.example` to
`.env` and fill it in:

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_MODEL=...
```

Without an API key the director won't run. `--program` still works and guesses each track's mood from the
audio.

## Tests

```bash
uv run pytest -q     # uses synthetic tracks, so you don't need any real music
```

## License

GPLv3, see [LICENSE](LICENSE). Only mix music you have the rights to.
