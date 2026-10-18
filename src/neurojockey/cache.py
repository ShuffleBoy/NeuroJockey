import hashlib
from pathlib import Path

from .config import CACHE_DIR

_CHUNK = 2 * 1024 * 1024


def track_id(path: str | Path) -> str:
    """Content-based id: size + head + tail of the file."""
    path = Path(path)
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(_CHUNK))
        if size > _CHUNK:
            f.seek(max(size - _CHUNK, _CHUNK))
            h.update(f.read(_CHUNK))
    return h.hexdigest()[:16]


def track_dir(tid: str) -> Path:
    d = CACHE_DIR / tid
    d.mkdir(parents=True, exist_ok=True)
    return d
