from __future__ import annotations

from pathlib import Path

import numpy as np

from silentspeak.config import Config
from silentspeak.dataset import (
    Segments,
    augment,
    collate,
    crop,
    feature_stats,
    prepare,
    split_mask,
    word_labels,
)
from silentspeak.features import FEATURE_DIM, FRAME_DIM, PRESENCE
from silentspeak.grid import read_alignments
from silentspeak.paths import landmarks_dir, raw_dir
from silentspeak.store import SpeakerLandmarks


def test_prepare_cuts_one_segment_per_aligned_word(corpus: Path, smoke_cfg: Config) -> None:
    seg = prepare(corpus, smoke_cfg)
    assert len(seg) == 6 * 12 * 6  # speakers x utterances x words
    assert set(seg.speakers) == {"s1", "s2", "s3", "s4", "s5", "s6"}
    assert seg.frames.shape[1] == FRAME_DIM
    assert "sil" not in set(seg.words)


def test_prepare_pads_each_word_by_two_frames(corpus: Path, smoke_cfg: Config) -> None:
    seg = prepare(corpus, smoke_cfg)
    align = read_alignments(raw_dir(corpus) / "alignments.zip", "s1")
    first = align[str(seg.utterances[0])][0]
    assert seg.words[0] == first.word
    assert len(seg.get(0)) == round(first.end - first.start) + 4


def test_prepare_drops_segments_without_a_face(
    tmp_path: Path, corpus: Path, smoke_cfg: Config
) -> None:
    lm = SpeakerLandmarks.load(landmarks_dir(corpus) / "s1.npz")
    lm.points[:] = np.nan
    lm_dir = tmp_path / "landmarks"
    lm.save(lm_dir / "s1.npz")
    (lm_dir / "s1.npz.partial.npz").unlink(missing_ok=True)
    seg = prepare(corpus, smoke_cfg, landmarks=lm_dir)
    assert len(seg) == 0


def test_segments_roundtrip(tmp_path: Path, corpus: Path, smoke_cfg: Config) -> None:
    seg = prepare(corpus, smoke_cfg)
    seg.save(tmp_path / "s.npz")
    back = Segments.load(tmp_path / "s.npz")
    np.testing.assert_array_equal(back.frames, seg.frames)
    np.testing.assert_array_equal(back.words, seg.words)


def test_speaker_splits_are_disjoint(corpus: Path, smoke_cfg: Config) -> None:
    seg = prepare(corpus, smoke_cfg)
    test = split_mask(seg, smoke_cfg, "test")
    val = split_mask(seg, smoke_cfg, "val")
    train = split_mask(seg, smoke_cfg, "train")
    assert not (test & val).any()
    assert (test | val | train).all()
    assert set(seg.speakers[test]) == {"s1", "s2"}
    assert set(seg.speakers[val]) == {"s3"}


def test_fewshot_words_get_no_label(corpus: Path, smoke_cfg: Config) -> None:
    seg = prepare(corpus, smoke_cfg)
    labels = word_labels(seg, smoke_cfg.trained_words)
    held = np.isin(seg.words, smoke_cfg.data.fewshot_words)
    assert (labels[held] == -1).all()
    assert (labels[~held] >= 0).all()


def test_augment_keeps_shape_and_presence_semantics(smoke_cfg: Config) -> None:
    rng = np.random.default_rng(0)
    frames = np.ones((14, FRAME_DIM), np.float32)
    for _ in range(20):
        out = augment(frames, smoke_cfg, rng)
        assert out.shape[1] == FRAME_DIM
        assert len(out) >= smoke_cfg.train.min_frames
        assert set(np.unique(out[:, PRESENCE])) <= {0.0, 1.0}
        assert np.all(out[out[:, PRESENCE] == 0] == 0)


def test_crop_is_random_in_training_and_central_otherwise() -> None:
    seq = np.arange(60, dtype=np.float32)[:, None]
    assert crop(seq, 50, None)[0, 0] == 5
    starts = {crop(seq, 50, np.random.default_rng(s))[0, 0] for s in range(30)}
    assert len(starts) > 1
    assert len(crop(seq[:10], 50, None)) == 10


def test_collate_pads_and_reports_lengths() -> None:
    import torch

    a = torch.ones(4, FEATURE_DIM)
    b = torch.ones(7, FEATURE_DIM)
    x, lengths, labels = collate([(a, 1), (b, 2)])
    assert x.shape == (2, 7, FEATURE_DIM)
    assert lengths.tolist() == [4, 7]
    assert labels.tolist() == [1, 2]
    assert float(x[0, 4:].abs().sum()) == 0


def test_feature_stats_leave_presence_unscaled(corpus: Path, smoke_cfg: Config) -> None:
    seg = prepare(corpus, smoke_cfg)
    mean, std = feature_stats(seg, smoke_cfg)
    assert mean.shape == (FEATURE_DIM,)
    assert float(mean[PRESENCE]) == 0.0
    assert float(std[PRESENCE]) == 1.0
    assert bool((std > 0).all())
