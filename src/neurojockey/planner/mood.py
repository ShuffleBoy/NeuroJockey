"""Track mood (dance-floor or sad), from audio features and an LLM that knows the songs."""

import json
import urllib.error

import numpy as np

from ..analysis import TrackAnalysis
from ..config import CACHE_DIR

CACHE = CACHE_DIR / "moods.json"
BATCH = 40

SYSTEM = """You are a club DJ and music curator who knows Russian and international pop, rap, phonk and
dance music. For each track (artist - title, with machine-measured tempo, key mode and energy) rate:
- valence: 0 = sad, melancholic, heartbroken or dark; 1 = happy, euphoric, party.
- club: how well it works on a club dance floor at peak time: 0 = sit-down / lyrical song, 1 = banger.
- mood: 1-3 words (e.g. "party", "melancholic", "aggressive", "romantic", "sad", "hype").
If you do not know a song, infer from the artist's style, the title and the audio numbers, and lower
your confidence. Reply with JSON matching the schema."""

SCHEMA = {
    "type": "object",
    "properties": {
        "tracks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "valence": {"type": "number"},
                    "club": {"type": "number"},
                    "mood": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["index", "valence", "club", "mood", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tracks"],
    "additionalProperties": False,
}


def audio_mood(t: TrackAnalysis) -> dict:
    major = 1.0 if t.key_mode == "major" else 0.0
    drive = float(np.clip((t.bpm - 95) / 45, 0, 1))
    valence = float(np.clip(0.25 + 0.2 * major + 0.35 * t.energy_level + 0.2 * drive, 0, 1))
    club = float(np.clip(0.15 + 0.6 * t.energy_level + 0.25 * drive, 0, 1))
    return {"valence": round(valence, 2), "club": round(club, 2), "mood": "", "src": "audio"}


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def attach_moods(tracks: list[TrackAnalysis], use_llm: bool = True, model: str | None = None,
                 log=lambda *a: None) -> None:
    """Fill `t.mood` for every track: the LLM's judgement blended with the audio guess."""
    from ..llm import ask, model_name

    cache = _load()
    todo = [t for t in tracks if t.track_id not in cache]
    if use_llm and todo:
        try:
            model = model_name(model)
        except RuntimeError as e:
            log(f"no mood LLM ({e}); using the audio guess")
            todo = []
        if todo:
            log(f"asking {model} about the mood of {len(todo)} tracks")
        for k in range(0, len(todo), BATCH):
            chunk = todo[k:k + BATCH]
            payload = {"tracks": [{"index": i, "track": t.title, "bpm": t.bpm, "mode": t.key_mode,
                                   "energy": t.energy_level} for i, t in enumerate(chunk)]}
            try:
                ans = ask(model, SYSTEM, payload, SCHEMA, "track_moods")
            except (urllib.error.URLError, RuntimeError, json.JSONDecodeError, KeyError) as e:
                log(f"mood LLM failed ({e}); using the audio guess")
                break
            for r in ans.get("tracks", []):
                if 0 <= r["index"] < len(chunk):
                    t = chunk[r["index"]]
                    cache[t.track_id] = {"title": t.title, "valence": r["valence"], "club": r["club"],
                                         "mood": r["mood"], "confidence": r["confidence"]}
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    for t in tracks:
        a = audio_mood(t)
        c = cache.get(t.track_id)
        if c:
            w = 0.5 + 0.4 * float(np.clip(c.get("confidence", 0.5), 0, 1))  # LLM weight 0.5..0.9
            t.mood = {"valence": round(w * c["valence"] + (1 - w) * a["valence"], 2),
                      "club": round(w * c["club"] + (1 - w) * a["club"], 2),
                      "mood": c["mood"], "src": "llm"}
        else:
            t.mood = a
