"""Clips for the web app's no-camera Replay mode.

Each clip is a short mouth-region frame strip (one JPEG sprite sheet) plus the
landmark sequence for the same frames, in the extraction format. The browser
runs the landmarks through exactly the same feature code and model as a live
capture, and draws the frames underneath.

``build_grid_clips`` cuts six held-out test-speaker words from GRID (CC BY 4.0,
credit shown in the app). Until the corpus is processed, ``build_synthetic_clips``
writes drawn placeholders with the same layout, marked ``"source": "synthetic"``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

from .config import Config
from .landmarks import N_LIPS, stored_position
from .synthetic import (
    INNER_LOWER,
    INNER_UPPER,
    OUTER_LOWER,
    OUTER_UPPER,
    MouthState,
    make_speaker,
    render_frames,
    word_states,
)
from .vocab import WORD_CATEGORY

F32 = NDArray[np.float32]
GRID_CREDIT = "GRID corpus, Cooke et al., CC BY 4.0"
SPRITE_COLUMNS = 8


def _round(a: NDArray[Any], digits: int = 5) -> Any:
    out = np.round(np.asarray(a, dtype=np.float64), digits)
    return [[None if math.isnan(v) else v for v in row] for row in out.reshape(len(out), -1)]


def mouth_crop(
    points: F32, frame_size: tuple[int, int], aspect: float, margin: float = 0.45
) -> tuple[int, int, int, int]:
    """Pixel box ``(x, y, w, h)`` around the lips over all frames, with the given aspect."""
    _, height = frame_size
    lips = points[:, :N_LIPS, :2].reshape(-1, 2) * height  # height units -> pixels
    lips = lips[np.isfinite(lips).all(axis=1)]
    if len(lips) == 0:
        raise ValueError("no face in any frame of the clip")
    x0, y0 = lips.min(axis=0)
    x1, y1 = lips.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = (x1 - x0) * (1 + 2 * margin)
    h = (y1 - y0) * (1 + 2 * margin)
    w = max(w, h * aspect)
    h = w / aspect
    x, y = round(cx - w / 2), round(cy - h / 2)
    return x, y, round(w), round(h)


def write_clip(
    out_dir: Path,
    clip_id: str,
    frames: list[Image.Image],
    points: F32,
    blend: F32,
    matrix: F32,
    crop: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    cfg: Config,
) -> dict[str, Any]:
    """Write the sprite sheet and landmark JSON; return the clip's index entry fields."""
    cw, ch = cfg.replay.crop_width, cfg.replay.crop_height
    cols = min(SPRITE_COLUMNS, len(frames))
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * cw, rows * ch), (20, 20, 20))
    x, y, w, h = crop
    for i, frame in enumerate(frames):
        tile = frame.crop((x, y, x + w, y + h)).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(tile, ((i % cols) * cw, (i // cols) * ch))
    sprite = f"{clip_id}.jpg"
    sheet.save(out_dir / sprite, quality=85, optimize=True, progressive=True)
    landmarks = f"{clip_id}.json"
    (out_dir / landmarks).write_text(
        json.dumps(
            {
                "frameSize": list(frame_size),
                "crop": list(crop),
                "points": _round(points),
                "blend": _round(blend, 4),
                "matrix": _round(matrix),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return {
        "sprite": sprite,
        "landmarks": landmarks,
        "frames": len(frames),
        "columns": cols,
        "frameWidth": cw,
        "frameHeight": ch,
    }


def _draw_synthetic(points: F32, frame_size: tuple[int, int]) -> Image.Image:
    """An illustrated mouth (not a photo) for placeholder clips."""
    w, h = frame_size
    img = Image.new("RGB", frame_size, (228, 196, 172))
    draw = ImageDraw.Draw(img)

    def contour(upper: tuple[int, ...], lower: tuple[int, ...]) -> list[tuple[float, float]]:
        ring = [*upper, *reversed(lower[1:-1])]
        return [
            (float(points[stored_position(i), 0] * h), float(points[stored_position(i), 1] * h))
            for i in ring
        ]

    if np.isfinite(points).all():
        draw.polygon(contour(OUTER_UPPER, OUTER_LOWER), fill=(176, 72, 88))
        draw.polygon(contour(INNER_UPPER, INNER_LOWER), fill=(58, 22, 30))
    del w
    return img.filter(ImageFilter.GaussianBlur(0.6))


def build_synthetic_clips(cfg: Config, out_dir: Path) -> dict[str, Any]:
    """Placeholder clips with the real layout, until GRID clips are built."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    frame_size = (360, 288)
    ctx = cfg.replay.context_frames
    clips = []
    for n, word in enumerate(cfg.replay.words, start=1):
        speaker = make_speaker(rng)
        core = word_states(word, 10, speaker.amplitude)
        states = [MouthState()] * ctx + core + [MouthState()] * ctx
        points, blend, matrix = render_frames(states, speaker, rng)
        crop = mouth_crop(points, frame_size, cfg.replay.crop_width / cfg.replay.crop_height)
        frames = [_draw_synthetic(p, frame_size) for p in points]
        clip_id = f"clip-{n}"
        entry = write_clip(out_dir, clip_id, frames, points, blend, matrix, crop, frame_size, cfg)
        pad = cfg.data.pad_frames
        clips.append(
            {
                "id": clip_id,
                "word": word,
                "category": WORD_CATEGORY[word],
                "speaker": "synthetic",
                "utterance": None,
                "fps": 25,
                "modelRange": [ctx - pad, ctx + len(core) + pad],
                **entry,
            }
        )
    index = {"source": "synthetic", "credit": None, "clips": clips}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def build_grid_clips(data_dir: Path, cfg: Config, out_dir: Path) -> dict[str, Any]:
    """Six held-out test-speaker words from GRID, landmarked with the pinned model."""
    from .extract import grid_clip_frames

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = cfg.replay.context_frames
    pad = cfg.data.pad_frames
    clips = []
    for n, word in enumerate(cfg.replay.words, start=1):
        speaker = cfg.replay.speakers[(n - 1) % len(cfg.replay.speakers)]
        got = grid_clip_frames(data_dir, speaker, word, ctx)
        crop = mouth_crop(
            got.points, got.frame_size, cfg.replay.crop_width / cfg.replay.crop_height
        )
        clip_id = f"clip-{n}"
        entry = write_clip(
            out_dir,
            clip_id,
            got.images,
            got.points,
            got.blend,
            got.matrix,
            crop,
            got.frame_size,
            cfg,
        )
        clips.append(
            {
                "id": clip_id,
                "word": word,
                "category": WORD_CATEGORY[word],
                "speaker": speaker,
                "utterance": got.utterance,
                "fps": 25,
                "modelRange": [max(0, got.word_start - pad), min(got.n, got.word_stop + pad)],
                **entry,
            }
        )
    index = {"source": "grid", "credit": GRID_CREDIT, "clips": clips}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index
