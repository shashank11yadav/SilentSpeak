"""A synthetic stand-in for GRID, in the exact on-disk formats of the real data.

Each word gets its own mouth-motion pattern (opening, width, pucker, smile over
time); each synthetic speaker has their own face size, position, head pose,
speaking rate and noise. That is enough for tests and ``silentspeak smoke`` to
exercise prepare -> train -> evaluate -> export with learnable signal, without
the 14 GB download. Nothing here is used for the real model.
"""

from __future__ import annotations

import io
import math
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .grid import TICKS_PER_FRAME
from .landmarks import (
    BLENDSHAPE_NAMES,
    CHIN,
    EYE_OUTER,
    N_BLENDSHAPES,
    N_STORED,
    NOSE_TIP,
    STORED_INDICES,
)
from .paths import landmarks_dir, raw_dir
from .store import SpeakerLandmarks
from .vocab import CATEGORIES

F32 = NDArray[np.float32]
UTTERANCE_FRAMES = 75

OUTER_UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_LOWER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
INNER_UPPER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
INNER_LOWER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
_ROW = {idx: i for i, idx in enumerate(STORED_INDICES)}


@dataclass(frozen=True)
class MouthState:
    open: float = 0.0
    width: float = 1.0
    pucker: float = 0.0
    smile: float = 0.0


def face_points(state: MouthState) -> NDArray[np.float64]:
    """Stored points of a frontal face in face units (eye distance 1, y up, z to viewer)."""
    pts = np.zeros((N_STORED, 3))
    half_w = 0.27 * state.width * (1 - 0.35 * state.pucker)
    cy = -0.45
    z0 = 0.05
    n = len(OUTER_UPPER) - 1
    for k in range(n + 1):
        theta = math.pi - k * math.pi / n
        c, s = math.cos(theta), math.sin(theta)
        lift = 0.03 * state.smile * c * c
        z = z0 + 0.06 * state.pucker * s
        pts[_ROW[OUTER_UPPER[k]]] = (half_w * c, cy + (0.05 + 0.04 * state.open) * s + lift, z)
        pts[_ROW[OUTER_LOWER[k]]] = (half_w * c, cy - (0.05 + 0.30 * state.open) * s + lift, z)
        pts[_ROW[INNER_UPPER[k]]] = (
            0.8 * half_w * c,
            cy + (0.004 + 0.03 * state.open) * s + lift,
            z - 0.01,
        )
        pts[_ROW[INNER_LOWER[k]]] = (
            0.8 * half_w * c,
            cy - (0.004 + 0.25 * state.open) * s + lift,
            z - 0.01,
        )
    pts[_ROW[EYE_OUTER[0]]] = (-0.5, 0.35, 0.0)
    pts[_ROW[EYE_OUTER[1]]] = (0.5, 0.35, 0.0)
    pts[_ROW[NOSE_TIP]] = (0.0, 0.0, 0.25)
    pts[_ROW[CHIN]] = (0.0, -0.85 - 0.3 * state.open, 0.0)
    return pts


def blendshapes(state: MouthState, rng: np.random.Generator) -> F32:
    b = np.abs(rng.normal(0, 0.01, size=N_BLENDSHAPES))
    names = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}
    b[names["jawOpen"]] = state.open
    b[names["mouthClose"]] = max(0.0, 0.1 - state.open)
    b[names["mouthPucker"]] = state.pucker
    b[names["mouthFunnel"]] = state.pucker * state.open
    for side in ("Left", "Right"):
        b[names["mouthSmile" + side]] = state.smile
        b[names["mouthStretch" + side]] = max(0.0, state.width - 1.0)
    return np.clip(b, 0, 1).astype(np.float32)


def rotation(yaw: float, pitch: float, roll: float) -> NDArray[np.float64]:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    out: NDArray[np.float64] = rz @ ry @ rx
    return out


@dataclass(frozen=True)
class Speaker:
    scale: float  # eye distance in image-height units
    centre: tuple[float, float]
    pose: tuple[float, float, float]
    rate: float
    amplitude: float
    noise: float


def make_speaker(rng: np.random.Generator) -> Speaker:
    return Speaker(
        scale=float(rng.uniform(0.18, 0.26)),
        centre=(float(rng.uniform(0.55, 0.75)), float(rng.uniform(0.35, 0.5))),
        pose=(
            float(rng.uniform(-0.25, 0.25)),
            float(rng.uniform(-0.15, 0.15)),
            float(rng.uniform(-0.1, 0.1)),
        ),
        rate=float(rng.uniform(0.85, 1.15)),
        amplitude=float(rng.uniform(0.8, 1.2)),
        noise=float(rng.uniform(0.002, 0.005)),
    )


def word_pattern(word: str) -> tuple[int, list[tuple[float, float, float]], float, float, float]:
    """Deterministic motion pattern: base length, opening bumps, width, pucker, smile."""
    rng = np.random.default_rng(zlib.crc32(word.encode()))
    length = int(rng.integers(5, 9)) + (len(word) > 1) * int(rng.integers(1, 5))
    bumps = [
        (float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.3, 1.0)), float(rng.uniform(0.1, 0.3)))
        for _ in range(int(rng.integers(1, 4)))
    ]
    return (
        length,
        bumps,
        float(rng.uniform(0.85, 1.2)),
        float(rng.uniform(0, 0.8)),
        float(rng.uniform(0, 0.8)),
    )


def word_states(word: str, n_frames: int, amplitude: float) -> list[MouthState]:
    _, bumps, width, pucker, smile = word_pattern(word)
    states = []
    for i in range(n_frames):
        t = (i + 0.5) / n_frames
        env = math.sin(math.pi * t)
        opening = sum(h * math.exp(-(((t - c) / s) ** 2)) for c, h, s in bumps)
        states.append(
            MouthState(
                open=min(1.0, amplitude * opening),
                width=1 + (width - 1) * env,
                pucker=pucker * env * amplitude,
                smile=smile * env,
            )
        )
    return states


def render_frames(
    states: list[MouthState], speaker: Speaker, rng: np.random.Generator, drop: float = 0.0
) -> tuple[F32, F32, F32]:
    """Image-space landmarks (height units), blendshapes and matrices for each state."""
    n = len(states)
    points = np.zeros((n, N_STORED, 3), np.float32)
    blend = np.zeros((n, N_BLENDSHAPES), np.float32)
    matrix = np.zeros((n, 4, 4), np.float32)
    flip = np.array([1.0, -1.0, -1.0])
    for i, state in enumerate(states):
        drift = 0.03 * math.sin(i / 9.0)
        r = rotation(speaker.pose[0] + drift, speaker.pose[1], speaker.pose[2])
        world = face_points(state) @ r.T
        img = world * flip * speaker.scale
        img[:, 0] += speaker.centre[0]
        img[:, 1] += speaker.centre[1]
        img += rng.normal(0, speaker.noise * speaker.scale, size=img.shape)
        points[i] = img
        blend[i] = blendshapes(state, rng)
        matrix[i] = np.eye(4)
        matrix[i, :3, :3] = r
        matrix[i, :3, 3] = (speaker.centre[0] * 30, -speaker.centre[1] * 30, -50)
        if rng.random() < drop:
            points[i] = np.nan
            blend[i] = np.nan
            matrix[i] = np.nan
    return points, blend, matrix


def utterance(
    speaker: Speaker, rng: np.random.Generator
) -> tuple[list[tuple[int, int, str]], list[MouthState]]:
    """One GRID-grammar sentence: alignment rows in frames and per-frame mouth states."""
    words = [str(rng.choice(ws)) for ws in CATEGORIES.values()]
    lengths = [max(3, round(word_pattern(w)[0] * speaker.rate)) for w in words]
    lead = int(rng.integers(6, 14))
    while lead + sum(lengths) > UTTERANCE_FRAMES - 4:
        lengths = [max(3, n - 1) for n in lengths]
    rows: list[tuple[int, int, str]] = [(0, lead, "sil")]
    states = [MouthState() for _ in range(lead)]
    cursor = lead
    for w, n in zip(words, lengths, strict=True):
        rows.append((cursor, cursor + n, w))
        states.extend(word_states(w, n, speaker.amplitude))
        cursor += n
    rows.append((cursor, UTTERANCE_FRAMES, "sil"))
    states.extend(MouthState() for _ in range(UTTERANCE_FRAMES - cursor))
    return rows, states


def alignment_text(rows: list[tuple[int, int, str]]) -> str:
    tick = int(TICKS_PER_FRAME)
    return "".join(f"{a * tick} {b * tick} {w}\n" for a, b, w in rows)


def write_corpus(
    data_dir: Path, speakers: list[str], utterances: int, seed: int, drop: float = 0.01
) -> None:
    """Write ``raw/alignments.zip``, ``landmarks/sN.npz`` and ``manifest.tsv``."""
    rng = np.random.default_rng(seed)
    raw = raw_dir(data_dir)
    raw.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("alignments/", "")
        for s in speakers:
            spk = make_speaker(rng)
            items = []
            for u in range(utterances):
                utt = f"syn{u:04d}"
                rows, states = utterance(spk, rng)
                zf.writestr(f"alignments/{s}/{utt}.align", alignment_text(rows))
                items.append((utt, *render_frames(states, spk, rng, drop=drop)))
            SpeakerLandmarks.from_utterances(items, frame_size=(360, 288)).save(
                landmarks_dir(data_dir) / f"{s}.npz"
            )
    (raw / "alignments.zip").write_bytes(buf.getvalue())
    size = (raw / "alignments.zip").stat().st_size
    (data_dir / "manifest.tsv").write_text(f"alignments.zip\tmd5:synthetic\t{size}\tsynthetic\n")
