"""Bar-level features, segmentation (intro/build/drop/break/outro) and highlight picking."""

import librosa
import numpy as np

from .model import Highlight, Segment

HOP = 512


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def _per_bar(frames: np.ndarray, frame_times: np.ndarray, t0: float, bar_len: float, n_bars: int):
    """Average framewise features (..., T) into bars -> (n_bars, ...)."""
    idx = np.floor((frame_times - t0) / bar_len).astype(int)
    out = []
    for b in range(n_bars):
        m = idx == b
        out.append(frames[..., m].mean(axis=-1) if m.any() else np.zeros(frames.shape[:-1]))
    return np.array(out)


def bar_features(mono: np.ndarray, sr: int, t0: float, bar_len: float, n_bars: int) -> dict:
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=HOP))
    P = S**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)

    rms_db = librosa.power_to_db(P.mean(axis=0))
    low_db = librosa.power_to_db(P[freqs < 150].mean(axis=0))
    mel = librosa.feature.melspectrogram(S=P, sr=sr, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    chroma = librosa.feature.chroma_stft(S=P, sr=sr)
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=sr)

    f = lambda x: _per_bar(x, times[: x.shape[-1]], t0, bar_len, n_bars)
    rms, low, ons = f(rms_db), f(low_db), f(onset)
    energy = np.clip(0.5 * _norm(rms) + 0.3 * _norm(low) + 0.2 * _norm(ons), 0, 1)
    return {
        "rms_db": rms, "energy": energy, "mfcc": f(mfcc), "chroma": f(chroma),
        "onset": ons, "low_db": low,
    }


def novelty(feats: dict, kernel: int = 8) -> np.ndarray:
    X = np.hstack([feats["mfcc"], 0.5 * feats["chroma"], 3 * feats["energy"][:, None]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    ssm = X @ X.T
    h = kernel // 2
    k = np.ones((kernel, kernel))
    k[:h, h:] = k[h:, :h] = -1
    n = len(ssm)
    pad = np.pad(ssm, h, mode="edge")
    nov = np.array([(pad[b : b + kernel, b : b + kernel] * k).sum() for b in range(n)])
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-9)


def _merge(segs: list[Segment], energy: np.ndarray) -> list[Segment]:
    out = [segs[0]]
    for s in segs[1:]:
        if s.label == out[-1].label:
            out[-1] = Segment(out[-1].start_bar, s.end_bar, s.label,
                              float(energy[out[-1].start_bar : s.end_bar].mean()))
        else:
            out.append(s)
    return out


def highlights(segs: list[Segment], energy: np.ndarray, vocal: np.ndarray | None,
               max_bars: int = 32) -> list[Highlight]:
    out = []
    for i, s in enumerate(segs):
        if s.label != "drop":
            continue
        prev = segs[i - 1].energy if i > 0 else s.energy
        voc = float(vocal[s.start_bar : s.end_bar].mean()) if vocal is not None else 0.0
        score = s.energy + 0.6 * (s.energy - prev) + 0.25 * voc + 0.1 * min(s.bars / 16, 1)
        kind = "chorus" if voc > 0.5 else "drop"
        out.append(Highlight(s.start_bar, min(s.end_bar, s.start_bar + max_bars), round(score, 3), kind))
    if not out:
        n, w = len(energy), min(16, len(energy))
        sums = [energy[b : b + w].mean() for b in range(0, max(1, n - w + 1), 4)]
        b = int(np.argmax(sums)) * 4
        out.append(Highlight(b, b + w, round(float(max(sums)), 3), "peak"))
    return sorted(out, key=lambda h: -h.score)


def vocal_presence(vocals_mono: np.ndarray, mix_mono: np.ndarray, sr: int, t0: float,
                   bar_len: float, n_bars: int) -> np.ndarray:
    def bar_db(x):
        rms = librosa.feature.rms(y=x, frame_length=2048, hop_length=HOP)
        t = librosa.frames_to_time(np.arange(rms.shape[1]), sr=sr, hop_length=HOP)
        return librosa.power_to_db(_per_bar(rms**2, t, t0, bar_len, n_bars)[:, 0])

    rel = bar_db(vocals_mono) - bar_db(mix_mono)
    return np.clip((rel + 18) / 12, 0, 1)


def stem_bar_db(stems: dict, sr: int, t0: float, bar_len: float, n_bars: int) -> dict[str, list[float]]:
    out = {}
    for name, x in stems.items():
        rms = librosa.feature.rms(y=x, frame_length=2048, hop_length=HOP)
        t = librosa.frames_to_time(np.arange(rms.shape[1]), sr=sr, hop_length=HOP)
        db = librosa.power_to_db(_per_bar(rms**2, t, t0, bar_len, n_bars)[:, 0], ref=1.0)
        out[name] = [round(float(v), 1) for v in db]
    return out


def beat_activity(x: np.ndarray, sr: int, t0: float, beat_len: float, n_beats: int) -> list[float]:
    """Per-beat activity 0..1 of a stem: its loudness relative to its own loud parts."""
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)
    idx = np.floor((t - t0) / beat_len).astype(int)
    db = np.full(n_beats, -100.0)
    for k in range(n_beats):
        m = idx == k
        if m.any():
            db[k] = 10 * np.log10(np.mean(rms[m] ** 2) + 1e-10)
    ref = np.percentile(db[db > -99], 85) if (db > -99).any() else 0
    return [round(float(v), 2) for v in np.clip((db - (ref - 24)) / 18, 0, 1)]


def eighth_db(x: np.ndarray, sr: int, t0: float, beat_len: float, n_beats: int) -> list[float]:
    """Vocal level per 8th note in dB relative to the stem's loud parts. Unlike `beat_activity` this is
    fine enough to see the breaths between rap lines - the only clean places to cut a vocal."""
    hop = 128
    rms = librosa.feature.rms(y=x, frame_length=512, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    idx = np.floor((t - t0) / (beat_len / 2)).astype(int)
    n = 2 * n_beats
    db = np.full(n, -60.0)
    ok = (idx >= 0) & (idx < n)
    e = np.zeros(n)
    c = np.zeros(n)
    np.add.at(e, idx[ok], rms[ok] ** 2)
    np.add.at(c, idx[ok], 1)
    m = c > 0
    db[m] = 10 * np.log10(e[m] / c[m] + 1e-10)
    ref = np.percentile(db[m], 85) if m.any() else 0.0
    return [round(float(v), 1) for v in np.clip(db - ref, -60, 6)]


def bar_chroma(tonal: np.ndarray, sr: int, t0: float, bar_len: float, n_bars: int) -> np.ndarray:
    """Per-bar CQT chroma (n_bars, 12), each row summing to 1. Feed it the harmonic stems
    (other + bass) when available: drums and rap vocals only smear it."""
    hop = 2048
    c = librosa.feature.chroma_cqt(y=tonal, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"),
                                   n_octaves=6, bins_per_octave=36)
    t = librosa.frames_to_time(np.arange(c.shape[1]), sr=sr, hop_length=hop)
    bars = _per_bar(c, t, t0, bar_len, n_bars)
    return bars / (bars.sum(axis=1, keepdims=True) + 1e-9)


def _recurrence(X: np.ndarray) -> np.ndarray:
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X @ X.T


def _seg_sim(R: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    """How well segment b repeats segment a: best mean of a recurrence diagonal (lag +-1 bar)."""
    la, lb = a[1] - a[0], b[1] - b[0]
    n = min(la, lb)
    if n < 4:
        return 0.0
    best = -1.0
    for off_a in range(0, la - n + 1, 4):
        for off_b in range(0, lb - n + 1, 4):
            for lag in (-1, 0, 1):
                i = np.arange(n) + a[0] + off_a
                j = np.arange(n) + b[0] + off_b + lag
                ok = (j >= 0) & (j < len(R))
                if ok.sum() >= 4:
                    best = max(best, float(R[i[ok], j[ok]].mean()))
    return best


def segment(feats: dict, chroma: np.ndarray, stem_db: dict | None, vocal: np.ndarray | None,
               min_len: int = 8) -> list[Segment]:
    """Sections from (1) timbre/harmony novelty, (2) instruments entering or leaving (stems) and
    (3) repetition: the loudest repeated vocal section is the chorus / hook, labelled "drop".

    Labels: intro, verse, build, drop (chorus / hook / main drop), break, outro."""
    energy = feats["energy"]
    n = len(energy)
    X = np.hstack([feats["mfcc"], 2.0 * chroma, 3 * energy[:, None]])
    R = _recurrence(X)
    nov = novelty({"mfcc": feats["mfcc"], "chroma": chroma, "energy": energy})

    def active(name, thr=12.0):
        if not stem_db or name not in stem_db:
            return None
        v = np.array(stem_db[name])
        return v > (np.percentile(v, 95) - thr)

    drums, bass, voc = active("drums"), active("bass"), active("vocals", 15.0)
    change = np.zeros(n)
    for act in (drums, bass, voc):
        if act is not None:
            a = act.astype(float)
            sm = np.convolve(a, np.ones(4) / 4, mode="same")
            change[1:] += np.abs(np.diff(sm, prepend=sm[0]))[1:]
    cands = []
    for b in range(4, n - 3, 4):
        jump = abs(energy[b:b + 4].mean() - energy[max(0, b - 4):b].mean())
        score = nov[b] + 1.2 * jump + 0.6 * change[b - 1:b + 2].max() + (0.12 if b % 8 == 0 else 0)
        cands.append((score, b))
    bounds = [0, n]
    for score, b in sorted(cands, reverse=True):
        if score < 0.4:
            break
        # short sections (4 bars) only at the edges or for a clear change (a 4-bar break, a build)
        short_ok = b < 12 or b > n - 12 or score >= 0.75
        if all(abs(b - x) >= (4 if short_ok else min_len) for x in bounds):
            bounds.append(b)
    bounds.sort()
    spans = list(zip(bounds, bounds[1:]))

    E = np.array([energy[s:e].mean() for s, e in spans])
    D = np.array([drums[s:e].mean() if drums is not None else 1.0 for s, e in spans])
    V = np.array([(vocal[s:e].mean() if vocal is not None else 0.5) for s, e in spans])

    # repetition groups
    k = len(spans)
    S = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            S[i, j] = S[j, i] = _seg_sim(R, spans[i], spans[j])
    group = list(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            if S[i, j] > 0.55 and abs(E[i] - E[j]) < 0.2:
                gi, gj = group[i], group[j]
                group = [gi if g == gj else g for g in group]
    reps = {g: [i for i in range(k) if group[i] == g] for g in set(group)}

    lo, hi = E.min(), E.max()
    rng_e = max(hi - lo, 1e-6)
    rel = (E - lo) / rng_e
    full = (D > 0.6) & (rel > 0.45)
    # chorus group: repeated, full band, loud, sung
    best_g, best_s = None, 0.0
    for g, idx in reps.items():
        if len(idx) < 2 or not all(full[i] for i in idx):
            continue
        s = float(np.mean(rel[idx])) + 0.3 * float(np.mean(V[idx])) + 0.1 * len(idx)
        if s > best_s:
            best_g, best_s = g, s
    labels = ["verse" if full[i] else "groove" for i in range(k)]
    if best_g is not None:
        for i in reps[best_g]:
            labels[i] = "drop"
    if hi - lo >= 0.12:
        # unrepeated but clearly the loudest full-band parts are drops too
        for i in range(k):
            if full[i] and rel[i] >= 0.8:
                labels[i] = "drop"
    if "drop" not in labels:
        labels[int(np.argmax(E))] = "drop"
    first_full = next((i for i in range(k) if full[i] or labels[i] == "drop"), 0)
    last_full = max(i for i in range(k) if full[i] or labels[i] == "drop")
    for i in range(k):
        if i < first_full:
            labels[i] = "intro"
        elif i > last_full:
            labels[i] = "outro"
        elif labels[i] in ("groove", "verse") and (D[i] < 0.4 or rel[i] < 0.3):
            labels[i] = "break"
    for i in range(1, k - 1):
        s, e = spans[i]
        if labels[i + 1] == "drop" and labels[i] in ("groove", "break", "verse") and e - s <= 8:
            if energy[e - 2:e].mean() > energy[s:s + 2].mean() + 0.05 or labels[i] == "break":
                labels[i] = "build"
    segs = [Segment(s, e, lab, float(energy[s:e].mean())) for (s, e), lab in zip(spans, labels)]
    return _merge(segs, energy)
