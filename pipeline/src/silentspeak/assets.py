"""The MediaPipe face landmarker model, downloaded once and pinned by SHA-256.

The web app fetches the same file (``web/scripts/fetch-assets.mjs`` pins the
same hash), so training landmarks and live landmarks come from one model.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path
from typing import Final

from .paths import PIPELINE_DIR

FACE_LANDMARKER_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_SHA256: Final[str] = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)
CACHE_DIR: Final[Path] = PIPELINE_DIR / ".cache"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def face_landmarker_model(cache_dir: Path = CACHE_DIR) -> Path:
    """Path to the verified model file, downloading it on first use."""
    path = cache_dir / "face_landmarker.task"
    if path.exists() and _digest(path) == FACE_LANDMARKER_SHA256:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(FACE_LANDMARKER_URL, timeout=60) as resp:
            tmp_path.write_bytes(resp.read())
        digest = _digest(tmp_path)
        if digest != FACE_LANDMARKER_SHA256:
            raise RuntimeError(f"face_landmarker.task hash mismatch: {digest}")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path
