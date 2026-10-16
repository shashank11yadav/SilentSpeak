"""Word segments cut from the extracted landmarks, speaker splits and batching."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self

import numpy as np
import torch
from numpy.typing import NDArray

from .config import Config
from .features import FRAME_DIM, PRESENCE, frames_features, resample, sequence_features
from .grid import read_alignments, speaker_sort_key, word_window
from .paths import landmarks_dir, raw_dir
from .store import SpeakerLandmarks
from .vocab import ALL_WORDS

log = logging.getLogger(__name__)

FEATURE_VERSION: Final[int] = 1
# Segments where the face was found in fewer frames than this are dropped.
MIN_PRESENT_FRACTION: Final[float] = 0.6
Split = Literal["train", "val", "test"]


@dataclass
class Segments:
    """Variable-length per-frame features, one row range per word segment."""

    frames: NDArray[np.float32]  # [total_frames, FRAME_DIM]
    offsets: NDArray[np.int64]  # [n + 1]
    words: NDArray[np.str_]
    speakers: NDArray[np.str_]
    utterances: NDArray[np.str_]

    def __len__(self) -> int:
        return len(self.words)

    def get(self, i: int) -> NDArray[np.float32]:
        return self.frames[self.offsets[i] : self.offsets[i + 1]]

    def subset(self, mask: NDArray[np.bool_]) -> Segments:
        idx = np.flatnonzero(mask)
        parts = [self.get(int(i)) for i in idx]
        lengths = [len(p) for p in parts]
        return Segments(
            frames=(np.concatenate(parts) if parts else np.zeros((0, FRAME_DIM), np.float32)),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            words=self.words[idx],
            speakers=self.speakers[idx],
            utterances=self.utterances[idx],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial.npz")
        np.savez_compressed(
            tmp,
            frames=self.frames,
            offsets=self.offsets,
            words=self.words,
            speakers=self.speakers,
            utterances=self.utterances,
            feature_version=np.array(FEATURE_VERSION),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path, allow_pickle=False) as z:
            if int(z["feature_version"]) != FEATURE_VERSION:
                raise ValueError(
                    f"{path} was prepared with another feature version; re-run prepare"
                )
            return cls(
                frames=z["frames"],
                offsets=z["offsets"],
                words=z["words"],
                speakers=z["speakers"],
                utterances=z["utterances"],
            )


def prepare(data_dir: Path, cfg: Config, landmarks: Path | None = None) -> Segments:
    """Cut every aligned word (plus padding) out of every extracted speaker."""
    lm_dir = landmarks or landmarks_dir(data_dir)
    files = sorted(lm_dir.glob("s*.npz"), key=lambda p: speaker_sort_key(p.stem))
    files = [f for f in files if f.stem[1:].isdigit()]
    if not files:
        raise FileNotFoundError(f"no extracted speakers in {lm_dir}; run `silentspeak extract`")
    align_zip = raw_dir(data_dir) / "alignments.zip"
    chunks: list[NDArray[np.float32]] = []
    words: list[str] = []
    speakers: list[str] = []
    utts: list[str] = []
    dropped: Counter[str] = Counter()
    for path in files:
        speaker = path.stem
        lm = SpeakerLandmarks.load(path)
        alignments = read_alignments(align_zip, speaker)
        feats = frames_features(lm.points, lm.blend, lm.matrix)
        for i, utt in enumerate(lm.utterances):
            if utt not in alignments:
                dropped["no alignment"] += 1
                continue
            a, b = int(lm.offsets[i]), int(lm.offsets[i + 1])
            utt_feats = feats[a:b]
            for w in alignments[utt]:
                if w.word not in ALL_WORDS:
                    dropped["unknown word"] += 1
                    continue
                start, stop = word_window(w, cfg.data.pad_frames, len(utt_feats))
                seg = utt_feats[start:stop]
                if len(seg) < cfg.train.min_frames:
                    dropped["too short"] += 1
                    continue
                if seg[:, PRESENCE].mean() < MIN_PRESENT_FRACTION:
                    dropped["face missing"] += 1
                    continue
                chunks.append(seg)
                words.append(w.word)
                speakers.append(speaker)
                utts.append(utt)
        log.info("%s: %d segments so far", speaker, len(words))
    if dropped:
        log.info("dropped segments: %s", dict(dropped))
    lengths = [len(c) for c in chunks]
    return Segments(
        frames=np.concatenate(chunks) if chunks else np.zeros((0, FRAME_DIM), np.float32),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        words=np.array(words, dtype=str),
        speakers=np.array(speakers, dtype=str),
        utterances=np.array(utts, dtype=str),
    )


def split_of(speaker: str, cfg: Config) -> Split:
    if speaker in cfg.data.test_speakers:
        return "test"
    if speaker in cfg.data.val_speakers:
        return "val"
    return "train"


def split_mask(segments: Segments, cfg: Config, split: Split) -> NDArray[np.bool_]:
    return np.array([split_of(str(s), cfg) == split for s in segments.speakers], dtype=bool)


def crop(
    seq: NDArray[np.float32], max_frames: int, rng: np.random.Generator | None
) -> NDArray[np.float32]:
    """Keep at most ``max_frames``: a random window in training, the centre otherwise."""
    if len(seq) <= max_frames:
        return seq
    extra = len(seq) - max_frames
    start = int(rng.integers(0, extra + 1)) if rng is not None else extra // 2
    return seq[start : start + max_frames]


def augment(
    frames: NDArray[np.float32], cfg: Config, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Training-time jitter that mimics webcam capture: speed, trimming, noise, drop-outs."""
    tc = cfg.train
    lo, hi = tc.time_stretch
    stretch = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    out = resample(frames, 25.0, 25.0 * stretch)
    # The live trimmer is not frame-exact; shave up to two frames off each end.
    a = int(rng.integers(0, 3))
    b = len(out) - int(rng.integers(0, 3))
    if b - a >= tc.min_frames:
        out = out[a:b]
    out = out.copy()
    present = out[:, PRESENCE] == 1.0
    noise = rng.normal(0, tc.feature_noise, size=(len(out), PRESENCE)).astype(np.float32)
    out[:, :PRESENCE] += noise * present[:, None]
    drop = rng.random(len(out)) < tc.frame_drop
    if drop.sum() < len(out) - tc.min_frames:
        out[drop] = 0.0
    return out


class WordDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        segments: Segments,
        labels: NDArray[np.int64],
        cfg: Config,
        train: bool,
        seed: int = 0,
    ) -> None:
        self.segments = segments
        self.labels = labels
        self.cfg = cfg
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        frames = self.segments.get(i)
        if self.train:
            frames = augment(frames, self.cfg, self.rng)
        seq = sequence_features(frames)
        seq = crop(seq, self.cfg.train.max_frames, self.rng if self.train else None)
        return torch.from_numpy(np.ascontiguousarray(seq)), int(self.labels[i])


def collate(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(x) for x, _ in batch])
    padded = torch.nn.utils.rnn.pad_sequence([x for x, _ in batch], batch_first=True)
    labels = torch.tensor([y for _, y in batch])
    return padded, lengths, labels


STATS_SAMPLE: Final[int] = 20_000


def feature_stats(segments: Segments, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean and std of model inputs over present training frames.

    Estimated from a fixed random sample of segments, which is plenty for 157
    means and deviations and keeps memory flat on the full corpus.
    """
    rng = np.random.default_rng(cfg.seed)
    n = len(segments)
    pick = rng.choice(n, size=STATS_SAMPLE, replace=False) if n > STATS_SAMPLE else range(n)
    seqs = [sequence_features(segments.get(int(i))) for i in pick]
    allf = np.concatenate(seqs) if seqs else np.zeros((1, 2 * FRAME_DIM - 1), np.float32)
    present = allf[:, PRESENCE] == 1.0
    sel = allf[present] if present.any() else allf
    mean = sel.mean(axis=0)
    std = sel.std(axis=0)
    # The presence flag stays as-is so a missing frame is an unambiguous zero.
    mean[PRESENCE] = 0.0
    std[PRESENCE] = 1.0
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def word_labels(segments: Segments, words: tuple[str, ...]) -> NDArray[np.int64]:
    index = {w: i for i, w in enumerate(words)}
    return np.array([index.get(str(w), -1) for w in segments.words], dtype=np.int64)
