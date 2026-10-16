from __future__ import annotations

import numpy as np

from silentspeak.evaluate import accepted_stats, calibrate_threshold, fewshot_episodes, softmax


def test_softmax_rows_sum_to_one() -> None:
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], np.float32))
    np.testing.assert_allclose(p.sum(1), [1, 1], rtol=1e-6)
    np.testing.assert_allclose(p[1], [1 / 3] * 3, rtol=1e-6)


class TestCalibrateThreshold:
    def test_returns_lowest_cut_meeting_the_target(self) -> None:
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        correct = np.array([1, 1, 1, 0, 0, 0], bool)
        # Accepting down to 0.6 gives 3/4 = 75%; down to 0.5 gives 60%.
        assert calibrate_threshold(scores, correct, 0.75, min_accepted=1) == 0.6

    def test_everything_accepted_when_all_right(self) -> None:
        scores = np.linspace(0.1, 0.9, 30)
        assert calibrate_threshold(scores, np.ones(30, bool), 0.75) == np.float64(0.1)

    def test_unreachable_target_accepts_only_the_top(self) -> None:
        scores = np.linspace(0.1, 0.9, 30)
        assert calibrate_threshold(scores, np.zeros(30, bool), 0.75) == 0.9

    def test_needs_enough_accepted_answers(self) -> None:
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        correct = np.array([1, 0, 0, 0], bool)
        assert calibrate_threshold(scores, correct, 0.75, min_accepted=2) == 0.9

    def test_empty_input(self) -> None:
        assert calibrate_threshold(np.zeros(0), np.zeros(0, bool), 0.75) == 1.0


def test_accepted_stats() -> None:
    precision, coverage = accepted_stats(
        np.array([0.9, 0.5, 0.2]), np.array([True, False, False]), 0.5
    )
    assert precision == 0.5
    assert coverage == 2 / 3
    assert accepted_stats(np.array([0.1]), np.array([True]), 0.5) == (None, 0.0)


def test_fewshot_is_perfect_on_separable_embeddings() -> None:
    rng = np.random.default_rng(0)
    centres = np.eye(4, 16, dtype=np.float32)
    words = np.array(["a", "b", "c", "d"] * 20)
    speakers = np.array(["s1"] * 40 + ["s2"] * 40)
    raw = centres[np.searchsorted(["a", "b", "c", "d"], words)] + rng.normal(0, 0.05, (80, 16))
    emb = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
    acc, tops, margins, right = fewshot_episodes(
        emb, words, speakers, ("a", "b", "c", "d"), shots=5, episodes=20, rng=rng
    )
    assert acc == 1.0
    assert right.all()
    assert (margins > 0.5).all()
    assert len(tops) == len(right) == 20 * 4 * 5


def test_fewshot_skips_speakers_with_too_few_takes() -> None:
    emb = np.eye(2, dtype=np.float32)
    acc, *_ = fewshot_episodes(
        emb,
        np.array(["a", "b"]),
        np.array(["s1", "s1"]),
        ("a", "b"),
        5,
        3,
        np.random.default_rng(0),
    )
    assert acc != acc  # NaN
