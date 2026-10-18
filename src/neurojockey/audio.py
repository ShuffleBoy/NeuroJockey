"""Audio I/O. Internally audio is float32, shape (channels=2, samples)."""

from pathlib import Path

import numpy as np
from pedalboard.io import AudioFile

from .config import SR

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}


def load(path: str | Path, sr: int = SR) -> np.ndarray:
    with AudioFile(str(path)).resampled_to(sr) as f:
        audio = f.read(f.frames)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return np.ascontiguousarray(audio[:2], dtype=np.float32)


def save(path: str | Path, audio: np.ndarray, sr: int = SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"quality": "V0"} if path.suffix.lower() == ".mp3" else {}
    with AudioFile(str(path), "w", sr, num_channels=audio.shape[0], **kwargs) as f:
        for a in range(0, audio.shape[1], 60 * sr):  # in blocks: no full-length copy of long mixes
            f.write(np.ascontiguousarray(audio[:, a:a + 60 * sr], dtype=np.float32))


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20))


def find_tracks(paths: list[Path], dedupe: bool = True, warn=None) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS and not f.name.startswith("."))
        elif p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return dedupe_tracks(out, warn) if dedupe else out


def _norm_name(path: Path) -> str:
    import re

    stem = re.sub(r"\(\d+\)$", "", path.stem.strip()).strip()
    return re.sub(r"\s+", " ", stem).lower()


def dedupe_tracks(paths: list[Path], warn=None) -> list[Path]:
    """Drop exact copies (same content) and "Song(1).m4a"-style duplicates of the same name.
    Identical audio saved under two *different* song names is a broken download: we cannot tell which
    name is right, so all of its copies are left out (and reported through `warn`)."""
    from collections import defaultdict

    from .cache import track_id

    by_id = defaultdict(list)
    for p in paths:
        by_id[track_id(p)].append(p)
    bad = set()
    for tid, group in by_id.items():
        names = {_norm_name(p) for p in group}
        if len(names) > 1:
            bad.add(tid)
            if warn:
                warn("same audio under different names, skipped: " + " = ".join(sorted(p.name for p in group)))
    seen_ids, seen_names, out = set(), set(), []
    for p in sorted(paths, key=lambda x: (len(x.stem), x.name)):  # prefer the name without "(1)"
        tid, name = track_id(p), _norm_name(p)
        if tid in bad or tid in seen_ids or name in seen_names:
            continue
        seen_ids.add(tid)
        seen_names.add(name)
        out.append(p)
    return sorted(out)
