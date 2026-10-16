"""Real GRID frames for the browser-versus-Python landmark parity check.

The synthetic parity fixture (``shared/fixtures/features.json``) proves that
``features.py`` and ``features.ts`` compute the same numbers from the same
inputs. It cannot prove that the browser *gets* the same inputs: MediaPipe's
JavaScript API returns the pose matrix as a flat array whose layout differs
from Python's ``[4, 4]`` array, and its web build runs a different inference
backend.

This module writes a handful of real frames from one held-out GRID speaker as
lossless PNGs, plus what the Python landmarker and ``frame_features`` make of
each one. The web end-to-end test runs the browser's landmarker on the same
PNGs, computes features with ``web/src/ml/features.ts``, and compares.

Both sides use IMAGE mode, so each frame is landmarked on its own and the
result does not depend on the frames before it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .assets import face_landmarker_model
from .extract import Landmarker
from .features import frame_features, to_height_units
from .landmarks import BLENDSHAPE_NAMES
from .video import SpeakerVideo

FIXTURE_NAME: Final[str] = "grid-frames.json"
FRAMES_DIR: Final[str] = "grid-frames"
JAW_OPEN: Final[int] = BLENDSHAPE_NAMES.index("jawOpen")

# The browser's MediaPipe (WebAssembly, CPU delegate) and Python's (XNNPACK on
# the CPU) are different builds of the same model. Measured on these frames in
# Chromium on an M4, they agree to 2e-5 in lip geometry, 3e-5 in the rotation
# and 2e-4 in blendshapes; the tolerances leave room for other CPUs. Reading
# the matrix with the wrong layout moves the geometry by about 0.012, so the
# check stays sharp enough to catch that.
TOLERANCE: Final[dict[str, float]] = {
    "geometry": 1e-3,  # lip geometry, eye-distance units (lip half-width is about 0.4)
    "blendshapes": 5e-3,  # blendshape scores and sums of two
    "rotation": 1e-3,  # each element of the pose matrix's rotation part
}
# The GPU delegate, which the live camera tries first, runs the network in
# reduced precision and was measured at 0.007 (geometry) and 0.06 (blendshapes).
GPU_TOLERANCE: Final[dict[str, float]] = {"geometry": 0.02, "blendshapes": 0.15}


def _nested(a: NDArray[Any]) -> Any:
    return np.asarray(a, dtype=np.float64).round(7).tolist()


def pick_frames(jaw_open: NDArray[np.float32], count: int) -> list[int]:
    """Frame indices spread over the range of mouth opening, closed to wide open."""
    order = np.argsort(jaw_open, kind="stable")
    picks = np.linspace(0, len(order) - 1, num=min(count, len(order)))
    return sorted({int(order[round(p)]) for p in picks})


def build_frame_fixture(
    zip_path: Path, speaker: str, out_dir: Path, utterance: str | None = None, count: int = 10
) -> dict[str, Any]:
    """Write ``<out_dir>/grid-frames/*.png`` and return the fixture contents."""
    frames_dir = out_dir / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    with SpeakerVideo(zip_path) as video:
        utt = utterance or video.utterances[0]
        rgbs = list(video.frames(utt))

    landmarker = Landmarker(face_landmarker_model(), video=False)
    try:
        found = [landmarker.detect_normalised(rgb) for rgb in rgbs]
    finally:
        landmarker.close()
    jaw = np.nan_to_num(np.array([b[JAW_OPEN] for _, b, _ in found], dtype=np.float32), nan=-1.0)

    frames = []
    for idx in pick_frames(jaw, count):
        name = f"{speaker}-{utt}-{idx:03d}.png"
        path = frames_dir / name
        Image.fromarray(rgbs[idx]).save(path, format="PNG", optimize=True)
        # PNG is lossless: the browser decodes exactly the pixels landmarked here.
        if not np.array_equal(np.asarray(Image.open(path).convert("RGB")), rgbs[idx]):
            raise RuntimeError(f"{path} does not round-trip")
        h, w = rgbs[idx].shape[:2]
        pts, blend, matrix = found[idx]
        frames.append(
            {
                "file": f"{FRAMES_DIR}/{name}",
                "frame": idx,
                "width": w,
                "height": h,
                "normalised": _nested(pts),
                "blend": _nested(blend),
                "matrix": _nested(matrix),
                "expected": _nested(frame_features(to_height_units(pts, w, h), blend, matrix)),
            }
        )

    return {
        "source": (
            f"GRID corpus (Cooke, Barker, Cunningham and Shao), speaker {speaker}, "
            f"utterance {utt}; CC BY 4.0, https://doi.org/10.5281/zenodo.3625687"
        ),
        "speaker": speaker,
        "utterance": utt,
        "runningMode": "IMAGE",
        "tolerance": TOLERANCE,
        "gpuTolerance": GPU_TOLERANCE,
        "frames": frames,
    }


def write_frame_fixture(
    zip_path: Path, speaker: str, out_dir: Path, utterance: str | None = None, count: int = 10
) -> Path:
    fixture = build_frame_fixture(zip_path, speaker, out_dir, utterance, count)
    path = out_dir / FIXTURE_NAME
    path.write_text(json.dumps(fixture) + "\n")
    return path
