"""Landmark extraction: the MediaPipe face landmarker over every GRID frame.

One ``sN.npz`` per speaker (see ``store.py``); a speaker whose file exists is
skipped, so an interrupted run resumes where it stopped. Speakers run in
parallel processes with ``--workers``. Only zips whose size matches
``manifest.tsv`` are considered, and each one's md5 is checked against it
before extraction.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from .assets import face_landmarker_model
from .features import to_height_units
from .grid import read_alignments, verify_download, video_speakers, word_window
from .landmarks import BLENDSHAPE_NAMES, N_BLENDSHAPES, N_STORED, STORED_INDICES
from .paths import raw_dir
from .store import SpeakerLandmarks
from .video import SpeakerVideo

log = logging.getLogger(__name__)
F32 = NDArray[np.float32]
FRAME_MS = 40  # 25 fps
UTTERANCE_GAP_MS = 1000
GRID_FRAMES_PER_UTTERANCE = 75


class Landmarker:
    """MediaPipe FaceLandmarker returning the stored subset.

    VIDEO mode (the default) tracks the face from frame to frame, as extraction
    does over an utterance; IMAGE mode treats every frame on its own, as the
    real-frame parity fixture does.
    """

    def __init__(self, model_path: Path, video: bool = True) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self._video = video
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._task = vision.FaceLandmarker.create_from_options(options)
        self._ts = 0
        self._checked_names = False

    def close(self) -> None:
        self._task.close()

    def gap(self) -> None:
        """Leave a pause in the timeline between two utterances."""
        self._ts += UTTERANCE_GAP_MS

    def detect(self, rgb: NDArray[np.uint8]) -> tuple[F32, F32, F32]:
        """Stored points in image-height units, blendshapes and the pose matrix."""
        h, w = rgb.shape[:2]
        pts, blend, matrix = self.detect_normalised(rgb)
        return to_height_units(pts, w, h), blend, matrix

    def detect_normalised(self, rgb: NDArray[np.uint8]) -> tuple[F32, F32, F32]:
        """As ``detect``, with the points left in MediaPipe's normalised units."""
        import mediapipe as mediapipe_lib

        image = mediapipe_lib.Image(
            image_format=mediapipe_lib.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        if self._video:
            self._ts += FRAME_MS
            result = self._task.detect_for_video(image, self._ts)
        else:
            result = self._task.detect(image)
        if not result.face_landmarks:
            return (
                np.full((N_STORED, 3), np.nan, np.float32),
                np.full(N_BLENDSHAPES, np.nan, np.float32),
                np.full((4, 4), np.nan, np.float32),
            )
        lms = result.face_landmarks[0]
        pts = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in STORED_INDICES], dtype=np.float32)
        cats = result.face_blendshapes[0]
        if not self._checked_names:
            names = tuple(c.category_name for c in cats)
            if names != BLENDSHAPE_NAMES:
                raise RuntimeError("unexpected blendshape order from MediaPipe")
            self._checked_names = True
        blend = np.array([c.score for c in cats], dtype=np.float32)
        matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
        return pts, blend, matrix.reshape(4, 4)


def extract_speaker(
    zip_path: Path,
    out_path: Path,
    model_path: Path,
    limit_utterances: int | None = None,
    progress: Any = None,
) -> dict[str, float]:
    """Landmark one speaker; returns frame count, seconds and missing-face count."""
    landmarker = Landmarker(model_path)
    items = []
    frame_size = (0, 0)
    frames_done, missing = 0, 0
    started = time.monotonic()
    try:
        with SpeakerVideo(zip_path) as video:
            utterances = video.utterances[:limit_utterances]
            for utt in utterances:
                pts, bls, mats = [], [], []
                for rgb in video.frames(utt):
                    frame_size = (rgb.shape[1], rgb.shape[0])
                    p, b, m = landmarker.detect(rgb)
                    pts.append(p)
                    bls.append(b)
                    mats.append(m)
                    missing += int(np.isnan(p[0, 0]))
                landmarker.gap()
                frames_done += len(pts)
                if pts:
                    items.append((utt, np.stack(pts), np.stack(bls), np.stack(mats)))
                if progress is not None:
                    progress.put(len(pts))
    finally:
        landmarker.close()
    seconds = time.monotonic() - started
    SpeakerLandmarks.from_utterances(items, frame_size).save(out_path)
    return {
        "frames": frames_done,
        "seconds": seconds,
        "missing": missing,
        "utterances": len(items),
    }


def _worker(args: tuple[str, Path, Path, Path, int | None, Any]) -> tuple[str, dict[str, float]]:
    speaker, zip_path, out_path, model_path, limit, progress = args
    return speaker, extract_speaker(zip_path, out_path, model_path, limit, progress)


def extract_all(
    data_dir: Path,
    out_dir: Path,
    speakers: list[str] | None = None,
    workers: int = 1,
    limit_utterances: int | None = None,
) -> dict[str, Any]:
    raw = raw_dir(data_dir)
    available = video_speakers(raw)
    wanted = speakers or available
    not_ready = [s for s in wanted if s not in available]
    if not_ready:
        log.warning("skipping speakers whose zip is missing or incomplete: %s", not_ready)
    todo = [s for s in wanted if s in available and not (out_dir / f"{s}.npz").exists()]
    done = [s for s in wanted if (out_dir / f"{s}.npz").exists()]
    if done:
        log.info("already extracted, skipping: %s", done)
    if not todo:
        return {"speakers": {}, "skipped": done, "not_ready": not_ready}
    for s in todo:
        verify_download(raw, f"{s}.zip")
    model_path = face_landmarker_model()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_speaker = limit_utterances or 1000
    total = len(todo) * per_speaker * GRID_FRAMES_PER_UTTERANCE
    results: dict[str, dict[str, float]] = {}
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager, tqdm(total=total, unit="frame", smoothing=0.05) as bar:
        progress = manager.Queue()
        jobs = [
            (s, raw / f"{s}.zip", out_dir / f"{s}.npz", model_path, limit_utterances, progress)
            for s in todo
        ]
        with ctx.Pool(processes=min(workers, len(jobs))) as pool:
            pending = pool.imap_unordered(_worker, jobs)
            remaining = len(jobs)
            while remaining:
                try:
                    while True:
                        bar.update(progress.get_nowait())
                except queue.Empty:
                    pass
                try:
                    speaker, stats = pending.next(timeout=0.5)
                except mp.TimeoutError:
                    continue
                results[speaker] = stats
                remaining -= 1
                log.info(
                    "%s: %d frames, %.1f ms/frame, %d without a face",
                    speaker,
                    stats["frames"],
                    1000 * stats["seconds"] / max(1, stats["frames"]),
                    stats["missing"],
                )
            while not progress.empty():
                bar.update(progress.get_nowait())
    wall = time.monotonic() - started
    frames = sum(r["frames"] for r in results.values())
    busy = sum(r["seconds"] for r in results.values())
    return {
        "speakers": results,
        "frames": frames,
        "wall_seconds": wall,
        "ms_per_frame_per_worker": 1000 * busy / max(1, frames),
        "ms_per_frame_wall": 1000 * wall / max(1, frames),
        "skipped": done,
        "not_ready": not_ready,
    }


@dataclass
class ClipFrames:
    utterance: str
    images: list[Image.Image]
    points: F32
    blend: F32
    matrix: F32
    frame_size: tuple[int, int]
    word_start: int  # clip-relative frame range of the word itself
    word_stop: int

    @property
    def n(self) -> int:
        return len(self.images)


def grid_clip_frames(data_dir: Path, speaker: str, word: str, context: int) -> ClipFrames:
    """The first utterance of ``speaker`` containing ``word``, cut to the word +- context."""
    raw = raw_dir(data_dir)
    if speaker not in video_speakers(raw):
        raise FileNotFoundError(f"{speaker}.zip is missing or still downloading")
    landmarker = Landmarker(face_landmarker_model())
    try:
        with SpeakerVideo(raw / f"{speaker}.zip") as video:
            alignments = read_alignments(raw / "alignments.zip", speaker, video.utterances)
            for utt in video.utterances:
                hit = next((w for w in alignments.get(utt, []) if w.word == word), None)
                if hit is None:
                    continue
                rgbs = list(video.frames(utt))
                lms = [landmarker.detect(rgb) for rgb in rgbs]
                start, stop = word_window(hit, 0, len(rgbs))
                a, b = max(0, start - context), min(len(rgbs), stop + context)
                return ClipFrames(
                    utterance=utt,
                    images=[Image.fromarray(r) for r in rgbs[a:b]],
                    points=np.stack([lm[0] for lm in lms[a:b]]),
                    blend=np.stack([lm[1] for lm in lms[a:b]]),
                    matrix=np.stack([lm[2] for lm in lms[a:b]]),
                    frame_size=(rgbs[0].shape[1], rgbs[0].shape[0]),
                    word_start=start - a,
                    word_stop=stop - a,
                )
    finally:
        landmarker.close()
    raise LookupError(f"{speaker} never says {word!r}")
