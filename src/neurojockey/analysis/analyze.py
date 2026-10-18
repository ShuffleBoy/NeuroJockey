"""Full track analysis pipeline with caching."""

from pathlib import Path

import numpy as np
import pyloudnorm
import soxr

from .. import stems as stemlib
from ..audio import load, to_mono
from ..cache import track_dir, track_id
from ..config import ANALYSIS_VERSION, SR
from . import structure
from .beats import fit_grid, refine_phase, track_beats
from .key import camelot, key_from_profile, key_name
from .model import TrackAnalysis

AN_SR = 22050
STRUCT_VERSION = 3


def analyze(path: str | Path, with_stems: bool = True, force: bool = False,
            log=lambda *a: None) -> TrackAnalysis:
    path = Path(path)
    tid = track_id(path)
    cache = track_dir(tid) / "analysis.json"
    if cache.exists() and not force:
        an = TrackAnalysis.load(cache)
        if an.version == ANALYSIS_VERSION and (an.has_stems or not with_stems):
            an.path = str(path)
            dirty = False
            if an.has_stems and an.vocal8 is None:  # added later: fill in from the cached stems
                an.vocal8 = _vocal8(tid, an)
                dirty = True
            if an.struct_version < STRUCT_VERSION:  # newer structure / key analysis, no re-tracking
                _upgrade_structure(an, path)
                dirty = True
            if dirty:
                an.save(cache)
            return an
        if an.version == ANALYSIS_VERSION:
            # analysed without stems (crate scan): keep the beat grid, add only what needs the stems
            an.path = str(path)
            log("stems")
            audio = load(path, SR)
            mono = soxr.resample(to_mono(audio), SR, AN_SR)
            st = _stem_features(an, path, audio, mono)
            _upgrade_structure(an, path, audio=audio, mono=mono, stems=st)
            an.save(cache)
            return an

    audio = load(path, SR)
    duration = audio.shape[1] / SR
    mono = soxr.resample(to_mono(audio), SR, AN_SR)

    log("beats")
    beats, downbeats = track_beats(mono, AN_SR)
    grid = fit_grid(beats, downbeats, duration)
    grid.t0 = refine_phase(mono, AN_SR, grid.bpm, grid.t0)
    if grid.t0 < -0.02:
        grid.t0 += 4 * 60 / grid.bpm
    bar_len = 4 * 60 / grid.bpm

    lufs = float(pyloudnorm.Meter(SR).integrated_loudness(audio.T))
    feats = structure.bar_features(mono, AN_SR, grid.t0, bar_len, grid.n_bars)

    # Cross-track energy: loudness + rhythmic density + tempo, measured in the loud parts.
    loud = feats["energy"] >= np.percentile(feats["energy"], 60)
    onset_density = float(feats["onset"][loud].mean())
    energy_level = float(np.clip(
        0.4 * np.clip((lufs + 16) / 10, 0, 1)
        + 0.35 * np.clip(onset_density / 3, 0, 1)
        + 0.25 * np.clip((grid.bpm - 90) / 60, 0, 1), 0, 1))

    # key / segments / highlights are filled in by _upgrade_structure below
    an = TrackAnalysis(
        path=str(path), track_id=tid, title=path.stem, duration=duration,
        bpm=grid.bpm, t0=grid.t0, n_bars=grid.n_bars, variable_tempo=grid.variable,
        grid_error_ms=round(grid.error_ms, 2),
        key="", key_pc=0, key_mode="", camelot="", key_confidence=0.0,
        lufs=round(lufs, 2), energy_level=round(energy_level, 3),
        bar_energy=[round(float(x), 3) for x in feats["energy"]], bar_vocal=None,
        segments=[], highlights=[],
        beats=[round(float(x), 4) for x in beats], downbeats=[round(float(x), 4) for x in downbeats],
        version=ANALYSIS_VERSION,
    )
    st = None
    if with_stems:
        log("stems")
        st = _stem_features(an, path, audio, mono)
    log("structure")
    _upgrade_structure(an, path, audio=audio, mono=mono, stems=st, feats=feats)
    an.save(cache)
    return an


def _stem_features(an: TrackAnalysis, path: Path, audio: np.ndarray, mono: np.ndarray) -> dict:
    """Separate (cached) and fill in the per-bar / per-beat stem features; returns the stems."""
    stemlib.separate(path, an.track_id, audio=audio)
    st = stemlib.load_stems(an.track_id)
    small = {k: soxr.resample(to_mono(v), SR, AN_SR) for k, v in st.items()}
    v = small["vocals"]
    bl = an.bar_len
    an.bar_vocal = [round(float(x), 3) for x in structure.vocal_presence(v, mono, AN_SR, an.t0, bl, an.n_bars)]
    an.bar_stem_db = structure.stem_bar_db(small, AN_SR, an.t0, bl, an.n_bars)
    an.beat_vocal = structure.beat_activity(v, AN_SR, an.t0, bl / 4, an.n_bars * 4)
    an.vocal8 = structure.eighth_db(v, AN_SR, an.t0, bl / 4, an.n_bars * 4)
    an.has_stems = True
    return st


def _vocal8(tid: str, an: TrackAnalysis) -> list[float]:
    st = stemlib.load_stems(tid, names=("vocals",))
    v = soxr.resample(to_mono(st["vocals"]), SR, AN_SR)
    return structure.eighth_db(v, AN_SR, an.t0, an.beat_len, an.n_bars * 4)


def _upgrade_structure(an: TrackAnalysis, path: Path, audio=None, mono=None, stems=None, feats=None) -> None:
    """Structure (repetition + stems) and key from the harmonic stems; uses cached stems."""
    if mono is None:
        if audio is None:
            audio = load(path, SR)
        mono = soxr.resample(to_mono(audio), SR, AN_SR)
    bar_len = an.bar_len
    rs = lambda x: soxr.resample(to_mono(x), SR, AN_SR)
    vocal = np.array(an.bar_vocal) if an.bar_vocal else None
    if an.has_stems and (stems is not None or stemlib.has_stems(an.track_id)):
        st = stems if stems is not None else stemlib.load_stems(an.track_id, names=("bass", "other"))
        tonal = rs(st["other"]) + 1.5 * rs(st["bass"])  # the bass names the root: helps major vs minor
    else:
        tonal = mono
    chroma = structure.bar_chroma(tonal, AN_SR, an.t0, bar_len, an.n_bars)
    weights = np.array(an.bar_energy) + 0.05
    pc, mode, conf = key_from_profile((chroma * weights[:, None]).sum(0))
    an.key, an.key_pc, an.key_mode, an.camelot = key_name(pc, mode), pc, mode, camelot(pc, mode)
    an.key_confidence = round(conf, 3)
    an.chroma_bars = [[round(float(v), 3) for v in row] for row in chroma]
    if feats is None:
        feats = structure.bar_features(mono, AN_SR, an.t0, bar_len, an.n_bars)
    an.segments = structure.segment(feats, chroma, an.bar_stem_db, vocal)
    an.highlights = structure.highlights(an.segments, feats["energy"], vocal)
    an.struct_version = STRUCT_VERSION
