from __future__ import annotations

import numpy as np
import pytest

from silentspeak.features import (
    FEATURE_DIM,
    FRAME_DIM,
    PRESENCE,
    frame_features,
    frames_features,
    mirror_frame,
    resample,
    sequence_features,
    to_height_units,
)
from silentspeak.landmarks import EYE_OUTER, N_BLENDSHAPES, N_STORED, stored_position

from .helpers import random_frame, rotation


def test_dimensions() -> None:
    assert FRAME_DIM == 79
    assert FEATURE_DIM == 2 * FRAME_DIM - 1
    assert PRESENCE == FRAME_DIM - 1


def test_present_frame_sets_presence_and_is_finite() -> None:
    points, blend, matrix = random_frame(np.random.default_rng(0))
    f = frame_features(points, blend, matrix)
    assert f.shape == (FRAME_DIM,)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()
    assert f[PRESENCE] == 1.0


@pytest.mark.parametrize("where", ["points", "matrix"])
def test_missing_face_gives_zeros_and_no_presence(where: str) -> None:
    points, blend, matrix = random_frame(np.random.default_rng(1))
    if where == "points":
        points[:] = np.nan
    else:
        matrix[:] = np.nan
    f = frame_features(points, blend, matrix)
    assert np.all(f == 0.0)


def test_partial_nan_counts_as_missing() -> None:
    points, blend, matrix = random_frame(np.random.default_rng(2))
    points[5, 1] = np.nan
    assert np.all(frame_features(points, blend, matrix) == 0.0)


def test_invariant_to_translation_and_scale() -> None:
    points, blend, matrix = random_frame(np.random.default_rng(3))
    a = frame_features(points, blend, matrix)
    b = frame_features(points * 1.7 + np.float32(0.2), blend, matrix)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_removes_head_rotation() -> None:
    rng = np.random.default_rng(4)
    points, blend, _ = random_frame(rng)
    # A face seen straight on, then the same face turned by r. Landmarks live in
    # image space (y down, z away), the matrix in a y-up, z-toward-viewer space.
    flip = np.diag([1.0, -1.0, -1.0])
    r = rotation(0.4, -0.2, 0.15)
    centred = points.astype(np.float64) - points.mean(axis=0)
    turned = (flip @ r @ flip @ centred.T).T + 0.5
    m0 = np.eye(4, dtype=np.float32)
    m1 = np.eye(4, dtype=np.float32)
    m1[:3, :3] = r * 1.3
    a = frame_features(points, blend, m0)
    b = frame_features(turned.astype(np.float32), blend, m1)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_mirrored_face_gives_identical_features() -> None:
    rng = np.random.default_rng(5)
    for _ in range(5):
        points, blend, matrix = random_frame(rng)
        mp, mb, mm = mirror_frame(points, blend, matrix, aspect=4 / 3)
        np.testing.assert_allclose(
            frame_features(points, blend, matrix), frame_features(mp, mb, mm), atol=1e-5
        )


def test_degenerate_eye_distance_counts_as_missing() -> None:
    points, blend, matrix = random_frame(np.random.default_rng(6))
    points[stored_position(EYE_OUTER[1])] = points[stored_position(EYE_OUTER[0])]
    assert np.all(frame_features(points, blend, matrix) == 0.0)


def test_to_height_units_scales_x_and_z_by_aspect() -> None:
    lm = np.array([[0.5, 0.25, -0.1]], dtype=np.float32)
    out = to_height_units(lm, width=640, height=480)
    np.testing.assert_allclose(out, [[0.5 * 4 / 3, 0.25, -0.1 * 4 / 3]], rtol=1e-6)


def test_sequence_features_appends_deltas_gated_by_presence() -> None:
    rng = np.random.default_rng(7)
    frames = [frame_features(*random_frame(rng)) for _ in range(4)]
    frames[2] = np.zeros(FRAME_DIM, dtype=np.float32)  # face lost in frame 2
    seq = sequence_features(np.stack(frames))
    assert seq.shape == (4, FEATURE_DIM)
    np.testing.assert_array_equal(seq[:, :FRAME_DIM], np.stack(frames))
    deltas = seq[:, FRAME_DIM:]
    assert np.all(deltas[0] == 0)
    np.testing.assert_allclose(deltas[1], frames[1][:PRESENCE] - frames[0][:PRESENCE], atol=1e-6)
    assert np.all(deltas[2] == 0)
    assert np.all(deltas[3] == 0)


def test_sequence_features_of_empty_sequence() -> None:
    assert sequence_features(np.zeros((0, FRAME_DIM), np.float32)).shape == (0, FEATURE_DIM)


class TestResample:
    def test_same_rate_is_identity(self) -> None:
        seq = np.arange(12, dtype=np.float32).reshape(6, 2)
        seq[:, 1] = 1.0
        np.testing.assert_array_equal(resample(seq, 25, 25), seq)

    def test_30_to_25_fps_interpolates_linearly(self) -> None:
        t = np.arange(31, dtype=np.float32) / 30.0
        seq = np.stack([t * 10, np.ones_like(t)], axis=1)
        out = resample(seq, 30, 25)
        assert out.shape == (26, 2)
        np.testing.assert_allclose(out[:, 0], np.arange(26) / 25.0 * 10, atol=1e-5)

    def test_duration_is_preserved_when_downsampling(self) -> None:
        seq = np.ones((61, 3), dtype=np.float32)
        assert resample(seq, 60, 25).shape[0] == 26

    def test_upsampling(self) -> None:
        seq = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        out = resample(seq, 12.5, 25)
        np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0])

    def test_does_not_blend_with_missing_frames(self) -> None:
        # Frame 1 has no face; a sample between frames 0 and 1 takes the nearer one.
        seq = np.array([[2.0, 1.0], [0.0, 0.0], [4.0, 1.0]], dtype=np.float32)
        out = resample(seq, 3, 4)  # samples at 0, 0.75, 1.5 source frames
        np.testing.assert_allclose(out, [[2.0, 1.0], [0.0, 0.0], [4.0, 1.0]])

    def test_empty_and_single_frame(self) -> None:
        assert resample(np.zeros((0, 4), np.float32), 30, 25).shape == (0, 4)
        one = np.ones((1, 4), np.float32)
        np.testing.assert_array_equal(resample(one, 30, 25), one)

    def test_rejects_bad_rates(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            resample(np.ones((3, 2), np.float32), 0, 25)


def test_blendshape_count_matches_mediapipe() -> None:
    assert N_BLENDSHAPES == 52
    assert N_STORED == 44


def test_vectorised_frames_features_match_the_per_frame_version() -> None:
    rng = np.random.default_rng(8)
    frames = [random_frame(rng) for _ in range(6)]
    points = np.stack([f[0] for f in frames])
    blend = np.stack([f[1] for f in frames])
    matrix = np.stack([f[2] for f in frames])
    points[2] = np.nan
    matrix[4, 0, 0] = np.nan
    expected = np.stack(
        [frame_features(p, b, m) for p, b, m in zip(points, blend, matrix, strict=True)]
    )
    np.testing.assert_allclose(frames_features(points, blend, matrix), expected, atol=1e-6)
    assert frames_features(points[:0], blend[:0], matrix[:0]).shape == (0, FRAME_DIM)
