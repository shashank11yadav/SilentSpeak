"""The real-frame fixture that the web end-to-end test compares the browser against."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from PIL import Image

from silentspeak.features import FRAME_DIM, PRESENCE, frame_features, to_height_units
from silentspeak.landmarks import BLENDSHAPE_NAMES, N_STORED
from silentspeak.paths import FIXTURES_DIR
from silentspeak.realframes import (
    FIXTURE_NAME,
    GPU_TOLERANCE,
    JAW_OPEN,
    TOLERANCE,
    pick_frames,
)

FIXTURE = FIXTURES_DIR / FIXTURE_NAME


def _load() -> dict[str, Any]:
    return dict(json.loads(FIXTURE.read_text()))


def _f32(x: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.asarray(x, dtype=np.float32)


def test_pick_frames_spans_closed_to_open() -> None:
    jaw = _f32([0.5, 0.1, 0.9, 0.3, 0.0, 0.7])
    picks = pick_frames(jaw, 3)
    assert picks == [2, 3, 4]  # widest, middle, closed, in frame order
    assert pick_frames(jaw, 10) == [0, 1, 2, 3, 4, 5]


def test_fixture_is_credited_and_complete() -> None:
    fx = _load()
    assert "CC BY 4.0" in fx["source"]
    assert fx["tolerance"] == TOLERANCE
    assert fx["gpuTolerance"] == GPU_TOLERANCE
    assert len(fx["frames"]) >= 8
    for f in fx["frames"]:
        assert (FIXTURES_DIR / f["file"]).stat().st_size < 150_000
        assert np.shape(f["normalised"]) == (N_STORED, 3)
        assert len(f["blend"]) == len(BLENDSHAPE_NAMES)
        assert np.shape(f["matrix"]) == (4, 4)
        assert len(f["expected"]) == FRAME_DIM
        assert f["expected"][PRESENCE] == 1.0
    jaw = [f["blend"][JAW_OPEN] for f in fx["frames"]]
    assert max(jaw) - min(jaw) > 0.2, "frames should cover closed and open mouths"


def test_expected_features_follow_from_the_stored_landmarks() -> None:
    for f in _load()["frames"]:
        pts = to_height_units(_f32(f["normalised"]), f["width"], f["height"])
        got = frame_features(pts, _f32(f["blend"]), _f32(f["matrix"]))
        np.testing.assert_allclose(got, _f32(f["expected"]), atol=1e-5)


def test_python_matrix_is_row_major_with_translation_in_the_last_column() -> None:
    for f in _load()["frames"]:
        m = np.asarray(f["matrix"])
        np.testing.assert_allclose(m[3], [0, 0, 0, 1], atol=1e-6)
        assert abs(m[2, 3]) > 1.0  # the face sits tens of units in front of the camera
        rot = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0).mean()
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-3)


@pytest.mark.mediapipe
def test_python_landmarker_reproduces_the_fixture() -> None:
    from silentspeak.assets import face_landmarker_model
    from silentspeak.extract import Landmarker

    landmarker = Landmarker(face_landmarker_model(), video=False)
    try:
        for f in _load()["frames"]:
            rgb = np.asarray(Image.open(FIXTURES_DIR / f["file"]).convert("RGB"), dtype=np.uint8)
            pts, blend, matrix = landmarker.detect_normalised(rgb)
            feats = frame_features(to_height_units(pts, f["width"], f["height"]), blend, matrix)
            # Same pixels and model; allow for MediaPipe builds on other platforms.
            np.testing.assert_allclose(feats, _f32(f["expected"]), atol=1e-3)
    finally:
        landmarker.close()
