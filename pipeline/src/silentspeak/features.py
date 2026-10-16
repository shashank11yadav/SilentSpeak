"""Per-frame lip features, deltas and frame-rate resampling.

``web/src/ml/features.ts`` mirrors this module line for line, and
``shared/fixtures/features.json`` (written by ``silentspeak export-fixtures``)
holds inputs and expected outputs that both test suites check to 1e-4.

Coordinate conventions:

* ``points`` are the stored face-mesh points (see ``landmarks.STORED_INDICES``)
  in *image-height units*: MediaPipe's normalised x and z multiplied by the
  image aspect ratio (width / height), y unchanged. That keeps the geometry
  isotropic whatever the camera's resolution.
* ``matrix`` is MediaPipe's 4x4 facial transformation matrix (row-major), which
  maps the canonical face into a y-up, z-toward-the-viewer camera space.

The features are identical for a face and its mirror image, so a mirrored
webcam feed and a non-mirrored training video look the same to the model.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from .landmarks import (
    BLEND_PAIRS,
    BLEND_SINGLE,
    BLENDSHAPE_NAMES,
    EYE_OUTER,
    LIP_MIDLINE,
    LIP_PAIRS,
    N_LIPS,
    N_STORED,
    blend_mirror_permutation,
    mirror_permutation,
    stored_position,
)

F32 = NDArray[np.float32]

GEOMETRY_DIM: Final[int] = 3 * len(LIP_PAIRS) + 2 * len(LIP_MIDLINE)  # 62
BLEND_DIM: Final[int] = len(BLEND_SINGLE) + len(BLEND_PAIRS)  # 16
FRAME_DIM: Final[int] = GEOMETRY_DIM + BLEND_DIM + 1  # 79, the last value is presence
PRESENCE: Final[int] = FRAME_DIM - 1
FEATURE_DIM: Final[int] = FRAME_DIM + PRESENCE  # frame features plus their deltas
FPS: Final[float] = 25.0

_PAIR_ROWS: Final = [(stored_position(a), stored_position(b)) for a, b in LIP_PAIRS]
_MID_ROWS: Final = [stored_position(m) for m in LIP_MIDLINE]
_EYE_ROWS: Final = (stored_position(EYE_OUTER[0]), stored_position(EYE_OUTER[1]))
_BLEND_SINGLE_IDX: Final = [BLENDSHAPE_NAMES.index(n) for n in BLEND_SINGLE]
_BLEND_PAIR_IDX: Final = [
    (BLENDSHAPE_NAMES.index(a), BLENDSHAPE_NAMES.index(b)) for a, b in BLEND_PAIRS
]
_MIN_EYE_DISTANCE: Final[float] = 1e-6


def to_height_units(landmarks: F32, width: int, height: int) -> F32:
    """Convert MediaPipe normalised landmarks ``[N, 3]`` to image-height units."""
    aspect = np.float32(width / height)
    out = np.array(landmarks, dtype=np.float32, copy=True)
    out[..., 0] *= aspect
    out[..., 2] *= aspect
    return out


def frame_features(points: F32, blend: F32, matrix: F32) -> F32:
    """Features for one frame: ``[N_STORED, 3]``, ``[52]``, ``[4, 4]`` -> ``[FRAME_DIM]``.

    A frame with no face (any NaN) or a degenerate eye distance gives all zeros,
    including the presence flag.
    """
    out = np.zeros(FRAME_DIM, dtype=np.float32)
    p = np.asarray(points, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if p.shape != (N_STORED, 3) or not (np.isfinite(p).all() and np.isfinite(m).all()):
        return out

    # Image space (y down, z away) -> the matrix's space (y up, z toward the viewer).
    v = p * np.array([1.0, -1.0, -1.0])
    rot = m[:3, :3]
    scale = float(np.mean(np.linalg.norm(rot, axis=0)))
    if scale <= 0:
        return out
    # Undo head rotation: row-vector form of R^T @ v.
    q = v @ (rot / scale)
    q = q - q[:N_LIPS].mean(axis=0)
    eye = float(np.linalg.norm(q[_EYE_ROWS[0]] - q[_EYE_ROWS[1]]))
    if eye < _MIN_EYE_DISTANCE:
        return out
    q = q / eye

    k = 0
    for ia, ib in _PAIR_ROWS:
        out[k] = (q[ia, 0] - q[ib, 0]) / 2
        out[k + 1] = (q[ia, 1] + q[ib, 1]) / 2
        out[k + 2] = (q[ia, 2] + q[ib, 2]) / 2
        k += 3
    for im in _MID_ROWS:
        out[k] = q[im, 1]
        out[k + 1] = q[im, 2]
        k += 2

    b = np.nan_to_num(np.asarray(blend, dtype=np.float64), nan=0.0)
    for i in _BLEND_SINGLE_IDX:
        out[k] = b[i]
        k += 1
    for ia, ib in _BLEND_PAIR_IDX:
        out[k] = b[ia] + b[ib]
        k += 1
    out[PRESENCE] = 1.0
    return out


def frames_features(points: F32, blend: F32, matrix: F32) -> F32:
    """Vectorised ``frame_features`` over ``[T, N, 3]``, ``[T, 52]``, ``[T, 4, 4]``."""
    t = len(points)
    out = np.zeros((t, FRAME_DIM), dtype=np.float32)
    if t == 0:
        return out
    p = np.asarray(points, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    ok = np.isfinite(p).all(axis=(1, 2)) & np.isfinite(m).all(axis=(1, 2))
    p = np.where(ok[:, None, None], p, 0.0)
    rot = np.where(ok[:, None, None], m[:, :3, :3], np.eye(3))
    scale = np.mean(np.linalg.norm(rot, axis=1), axis=1)
    ok &= scale > 0
    scale = np.where(ok, scale, 1.0)
    v = p * np.array([1.0, -1.0, -1.0])
    q = np.einsum("tni,tij->tnj", v, rot / scale[:, None, None])
    q = q - q[:, :N_LIPS].mean(axis=1, keepdims=True)
    eye = np.linalg.norm(q[:, _EYE_ROWS[0]] - q[:, _EYE_ROWS[1]], axis=1)
    ok &= eye >= _MIN_EYE_DISTANCE
    q = q / np.where(ok, eye, 1.0)[:, None, None]

    ia = [a for a, _ in _PAIR_ROWS]
    ib = [b for _, b in _PAIR_ROWS]
    geo = np.empty((t, len(_PAIR_ROWS), 3))
    geo[:, :, 0] = (q[:, ia, 0] - q[:, ib, 0]) / 2
    geo[:, :, 1:] = (q[:, ia, 1:] + q[:, ib, 1:]) / 2
    mid = q[:, _MID_ROWS, 1:]
    b = np.nan_to_num(np.asarray(blend, dtype=np.float64), nan=0.0)
    single = b[:, _BLEND_SINGLE_IDX]
    pairs = b[:, [i for i, _ in _BLEND_PAIR_IDX]] + b[:, [j for _, j in _BLEND_PAIR_IDX]]
    feats = np.concatenate(
        [geo.reshape(t, -1), mid.reshape(t, -1), single, pairs, np.ones((t, 1))], axis=1
    )
    out[ok] = feats[ok].astype(np.float32)
    return out


def sequence_features(frames: F32) -> F32:
    """Append frame-to-frame deltas: ``[T, FRAME_DIM]`` -> ``[T, FEATURE_DIM]``.

    A delta is zero for the first frame and wherever either frame has no face.
    """
    frames = np.asarray(frames, dtype=np.float32)
    deltas = np.zeros((len(frames), PRESENCE), dtype=np.float32)
    if len(frames) > 1:
        gate = frames[1:, PRESENCE] * frames[:-1, PRESENCE]
        deltas[1:] = (frames[1:, :PRESENCE] - frames[:-1, :PRESENCE]) * gate[:, None]
    return np.concatenate([frames, deltas], axis=1)


def resample(seq: F32, src_fps: float, dst_fps: float = FPS) -> F32:
    """Resample frame features ``[T, FRAME_DIM]`` from ``src_fps`` to ``dst_fps``.

    Samples are linear interpolations between neighbouring source frames, except
    that a sample next to a frame with no face takes the nearer frame instead of
    blending a real frame with zeros.
    """
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"fps must be positive, got {src_fps} -> {dst_fps}")
    seq = np.asarray(seq, dtype=np.float32)
    n = len(seq)
    if n <= 1:
        return seq.copy()
    n_out = int(np.floor((n - 1) * dst_fps / src_fps + 1e-9)) + 1
    positions = np.arange(n_out, dtype=np.float64) * (src_fps / dst_fps)
    return interpolate_at(seq, positions)


def interpolate_at(seq: F32, positions: NDArray[np.float64]) -> F32:
    """Sample ``seq`` at fractional frame positions (see ``resample``)."""
    n = len(seq)
    pos = np.clip(positions, 0, n - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    w = pos - i0
    present = (seq[i0, -1] == 1.0) & (seq[i1, -1] == 1.0)
    w = np.where(present, w, np.where(w < 0.5, 0.0, 1.0))
    out = seq[i0] * (1 - w[:, None]) + seq[i1] * w[:, None]
    return out.astype(np.float32)


def mirror_frame(points: F32, blend: F32, matrix: F32, aspect: float) -> tuple[F32, F32, F32]:
    """What the landmarker would report for the horizontally mirrored image.

    Used to test mirror invariance and to build the parity fixture.
    """
    p = np.asarray(points, dtype=np.float32)[mirror_permutation()].copy()
    p[:, 0] = np.float32(aspect) - p[:, 0]
    b = np.asarray(blend, dtype=np.float32)[blend_mirror_permutation()].copy()
    flip = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    m = flip @ np.asarray(matrix, dtype=np.float32) @ flip
    return p, b, m
