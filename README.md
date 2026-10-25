# NeuroJockey

[![tests](https://github.com/ShuffleBoy/NeuroJockey/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/ShuffleBoy/NeuroJockey/actions/workflows/tests.yml)
[![version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FShuffleBoy%2FNeuroJockey%2Fmaster%2Fpyproject.toml&query=%24.project.version&label=version)](pyproject.toml)
[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

An offline AI DJ. Give it a folder of tracks and it renders one beatmatched mix. It finds the beats, splits each
track into vocals, drums, bass and the rest, then picks the order and a transition for every pair.

Most of the code was written by AI. A human listened to the mixes and complained until there was nothing left
to complain about.

## Install

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and `ffmpeg` (ffmpeg is only used for `.m4a` files).

```bash
git clone https://github.com/ShuffleBoy/NeuroJockey && cd NeuroJockey
uv sync
```

The first sync is large: it installs PyTorch, Demucs, beat_this, librosa and pedalboard, and the models
download on the first run. On Apple Silicon they run on MPS. Anywhere else they run on the CPU, which is slower.

## Usage

```bash
uv run nj mix tracks                                 # highlights mix → out/mix.mp3
uv run nj mix tracks -p club -d 60 -o out/club.mp3   # one-hour club set
uv run nj render out/club.set.json                   # re-render after editing the set
```

`tracks` is any folder of music, subfolders included. `nj` and `neurojockey` are the same command.

| option | default | |
|---|---|---|
| `-p, --preset` | `highlights` | kind of mix, see below |
| `-d, --duration` | all tracks / 60 | target length in minutes |
| `-o` | `out/mix.mp3` | output file: `.wav`, `.mp3` or `.flac` |

### Presets

| preset | tracks | structure |
|---|---|---|
| `full` | all, start to finish | best-flowing order |
| `highlights` | all, best 48 bars of each | best-flowing order |
| `club` | picked from the folder | warm-up → dance → breather → peak → closing |
| `warmup` | picked from the folder | opening → groove |
| `peak` | picked from the folder | peak → release |

### Output

Next to the audio file `mix` writes:

- `.set.json` is the plan: tracks, regions and transitions. Edit it and run `nj render` to re-render.
- `.md` is the tracklist plus a QA report for every transition.

The first analysis of a track takes about 25 s. Analyses and stems are cached in `~/.cache/neurojockey/`
(move it with `NEUROJOCKEY_CACHE`).

## LLM (optional)

The `club`, `warmup` and `peak` presets can ask a language model which tracks are dance-floor material and
which are sad songs about an ex. Any OpenAI-compatible API works (OpenAI, OpenRouter, a local server).
Copy `.env.example` to `.env`:

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_MODEL=...
```

Without a key, it guesses the mood from the audio.

## Development

```bash
uv run pytest -q   # synthetic tracks, no real music needed
```

## License

GPLv3, see [LICENSE](LICENSE). Only mix music you have the rights to.
