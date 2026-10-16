"""Builders for synthetic landmark frames shared by the tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from silentspeak.landmarks import N_BLENDSHAPES, N_STORED

F32 = NDArray[np.float32]


def rotation(yaw: float, pitch: float, roll: float) -> NDArray[np.float64]:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    out: NDArray[np.float64] = rz @ ry @ rx
    return out


def random_frame(rng: np.random.Generator) -> tuple[F32, F32, F32]:
    points = rng.normal(0.5, 0.05, size=(N_STORED, 3)).astype(np.float32)
    blend = rng.uniform(0, 1, size=N_BLENDSHAPES).astype(np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation(*rng.uniform(-0.3, 0.3, size=3)) * rng.uniform(0.8, 1.2)
    matrix[:3, 3] = rng.normal(0, 5, size=3)
    return points, blend, matrix
