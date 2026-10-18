"""Source separation into drums / bass / other / vocals with Demucs (cached)."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import load
from .cache import track_dir
from .config import SR, STEM_NAMES

DEFAULT_MODEL = "htdemucs"


def stem_dir(tid: str, model: str = DEFAULT_MODEL) -> Path:
    return track_dir(tid) / f"stems_{model}"


def has_stems(tid: str, model: str = DEFAULT_MODEL) -> bool:
    d = stem_dir(tid, model)
    return all((d / f"{n}.flac").exists() for n in STEM_NAMES)


@lru_cache(maxsize=2)
def _model(name: str):
    from demucs.pretrained import get_model

    net = get_model(name)
    net.eval()
    return net


def separate(path: str | Path, tid: str, model: str = DEFAULT_MODEL, audio: np.ndarray | None = None) -> Path:
    out = stem_dir(tid, model)
    if has_stems(tid, model):
        return out
    import torch  # imported lazily: render workers only read cached stems
    from demucs.apply import apply_model

    net = _model(model)
    if audio is None or net.samplerate != SR:
        audio = load(path, net.samplerate)
    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with torch.no_grad():
        try:
            sources = apply_model(net, wav[None], device=device, split=True, overlap=0.25, progress=True)[0]
        except Exception:
            sources = apply_model(net, wav[None], device="cpu", split=True, overlap=0.25, progress=True)[0]
    sources = sources * std + mean
    out.mkdir(parents=True, exist_ok=True)
    for name, src in zip(net.sources, sources):
        if name in STEM_NAMES:
            sf.write(out / f"{name}.flac", src.cpu().numpy().T, net.samplerate, subtype="PCM_24")
    return out


def load_stems(tid: str, model: str = DEFAULT_MODEL, names=STEM_NAMES) -> dict[str, np.ndarray]:
    d = stem_dir(tid, model)
    out = {}
    for n in names:
        data, sr = sf.read(d / f"{n}.flac", dtype="float32", always_2d=True)
        assert sr == SR, f"stem sample rate {sr} != {SR}"
        out[n] = np.ascontiguousarray(data.T)
    return out
