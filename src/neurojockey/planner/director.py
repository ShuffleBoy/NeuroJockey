"""LLM set director: picks the track order and transitions, checked by the rule planner."""

import json
import urllib.error

from ..analysis import TrackAnalysis
from ..engine.plan import SetList
from ..llm import ask, model_name
from ..transitions import REGISTRY
from .compat import best_ratio
from .planner import PlanOptions, plan_set

SYSTEM = """You are an experienced club DJ planning a mix. You get machine analysis of each track
(tempo, Camelot key, energy, structure, vocal presence) and a catalog of transitions the mixing
engine can perform. Choose the track order and one transition per adjacent pair.

Guidelines:
- Build a journey: warm up, build to a peak around 70-80% of the set, then release a little.
- Prefer harmonic neighbours on the Camelot wheel (same number, +-1, or relative major/minor).
- Beatmatched transitions work up to a ~16% tempo gap (both decks meet in the middle; half/double
  time counts), but vocals stretched more than ~8% each sound off - above ~10% prefer a clean break
  (reverb_wash first; cut).
- Stem transitions need stems on both tracks; acapella/mashup need a vocal on the outgoing track
  and very compatible keys.
- Vary techniques; do not repeat the same transition back to back unless it clearly fits best.
- Mostly smooth, musical techniques (bass_swap, drum_bed, blend, stem_relay, reverb_wash); keep showy
  effects (loop_roll, riser) rare - a pro set uses them once at most, and loop_roll
  does not suit vocal tracks.
- Two lead vocals must never fight; when both tracks are vocal-heavy prefer drum_bed or acapella.
- Energy going up suits a bass_swap or cut into a drop; energy going down suits reverb_wash or a
  long blend.
In each reason call the outgoing track "A" and the incoming one "B" - never refer to tracks by index.
Reply with JSON matching the schema."""


def _track_summary(i: int, t: TrackAnalysis) -> dict:
    h = t.highlights[0]
    return {
        "index": i,
        "title": t.title,
        "bpm": t.bpm,
        "camelot": t.camelot,
        "energy": t.energy_level,
        "has_stems": t.has_stems,
        "duration_min": round(t.duration / 60, 1),
        "structure": [f"{s.label} {s.start_bar}-{s.end_bar}" for s in t.segments],
        "vocal_share": round(t.vocal_amount(0, t.n_bars), 2),
        "best_highlight": f"{h.kind} bars {h.start_bar}-{h.end_bar}",
    }


def _schema(names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "order": {"type": "array", "items": {"type": "integer"},
                      "description": "track indices in play order, each exactly once"},
            "transitions": {
                "type": "array",
                "description": "one entry per adjacent pair in the chosen order",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": names},
                        "reason": {"type": "string"},
                    },
                    "required": ["type", "reason"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string", "description": "short description of the set's arc"},
        },
        "required": ["order", "transitions", "notes"],
        "additionalProperties": False,
    }


def llm_plan(tracks: list[TrackAnalysis], opts: PlanOptions, model: str | None = None) -> SetList:
    model = model_name(model)
    names = opts.transitions or list(REGISTRY)
    catalog = [{"name": n, "title": REGISTRY[n].title, "description": REGISTRY[n].description,
                "needs_stems": REGISTRY[n].needs_stems, "beatmatched": REGISTRY[n].beatmatched}
               for n in names]
    payload = {
        "mode": opts.mode,
        "tracks": [_track_summary(i, t) for i, t in enumerate(tracks)],
        "transitions": catalog,
        "tempo_gap_between_tracks": {f"{i}->{j}": round(best_ratio(a.bpm, b.bpm)[1], 3)
                                     for i, a in enumerate(tracks) for j, b in enumerate(tracks) if i != j},
    }
    if opts.keep_order:
        payload["constraint"] = "Keep the given track order; only choose transitions."

    try:
        data = ask(model, SYSTEM, payload, _schema(names), "dj_set")
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError, KeyError) as e:
        detail = e.read().decode()[:300] if isinstance(e, urllib.error.HTTPError) else e
        opts.log(f"[yellow]LLM director failed ({detail}); using rules[/]")
        return plan_set(tracks, opts)

    order = data.get("order", [])
    if opts.keep_order or sorted(order) != list(range(len(tracks))):
        order = list(range(len(tracks)))
    trans = data.get("transitions", [])[: len(tracks) - 1]
    if len(trans) < len(tracks) - 1 or any(t.get("type") not in REGISTRY for t in trans):
        opts.log("[yellow]LLM returned an incomplete plan; using rules[/]")
        return plan_set(tracks, opts)

    opts.keep_order = True
    opts.forced = [t["type"] for t in trans]
    sl = plan_set([tracks[i] for i in order], opts)
    for st, t in zip(sl.transitions, trans):
        if st.type == t["type"]:
            st.reason = f"{t['reason']} — {st.reason.replace(' [forced]', '')}"
    sl.notes = f"[{model}] {data.get('notes', '')}"
    return sl
