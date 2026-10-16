"""The shared parity fixture: Python must reproduce what the web tests assert against."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from silentspeak.export import build_fixture
from silentspeak.features import (
    frame_features,
    frames_features,
    resample,
    sequence_features,
)
from silentspeak.landmarks import BLENDSHAPE_NAMES, STORED_INDICES
from silentspeak.paths import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "features.json"


def _arr(x: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.array(x, dtype=np.float64).astype(np.float32)


def _load() -> dict[str, Any]:
    return dict(json.loads(FIXTURE.read_text()))


def test_committed_fixture_is_up_to_date() -> None:
    fresh = json.loads(json.dumps(build_fixture()))
    assert _load() == fresh, "re-run `uv run silentspeak export-fixtures`"


def test_fixture_constants_match_the_code() -> None:
    fx = _load()
    assert fx["storedIndices"] == list(STORED_INDICES)
    assert fx["blendshapeNames"] == list(BLENDSHAPE_NAMES)


def test_python_matches_fixture_frames() -> None:
    fx = _load()
    for case in fx["frames"]:
        got = frame_features(_arr(case["points"]), _arr(case["blend"]), _arr(case["matrix"]))
        np.testing.assert_allclose(got, _arr(case["expected"]), atol=fx["tolerance"])


def test_mirrored_frame_has_the_same_expected_features() -> None:
    fx = _load()
    cases = {c["name"]: c for c in fx["frames"]}
    assert cases["open-mirrored"]["expected"] == cases["open"]["expected"]
    assert cases["open-mirrored"]["points"] != cases["open"]["points"]


def test_python_matches_fixture_sequence_including_mirror_and_resampling() -> None:
    fx = _load()
    seq = fx["sequence"]
    for names in (
        ("points", "blend", "matrix"),
        ("mirroredPoints", "mirroredBlend", "mirroredMatrix"),
    ):
        feats = frames_features(*(_arr(seq[n]) for n in names))
        np.testing.assert_allclose(feats, _arr(seq["frameFeatures"]), atol=fx["tolerance"])
        out = sequence_features(resample(feats, seq["srcFps"], seq["dstFps"]))
        np.testing.assert_allclose(out, _arr(seq["expected"]), atol=fx["tolerance"])
    assert len(seq["expected"]) < len(seq["points"])  # 30 fps -> 25 fps
