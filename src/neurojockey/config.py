import os
from pathlib import Path

SR = 44100
CACHE_DIR = Path(os.environ.get("NEUROJOCKEY_CACHE", Path.home() / ".cache" / "neurojockey"))
STEM_NAMES = ("drums", "bass", "other", "vocals")
TARGET_LUFS = -11.0
ANALYSIS_VERSION = 3
