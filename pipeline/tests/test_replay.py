from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from silentspeak.config import Config
from silentspeak.features import frames_features
from silentspeak.replay import build_synthetic_clips, mouth_crop


def test_synthetic_clips_have_the_web_layout(tmp_path: Path, smoke_cfg: Config) -> None:
    index = build_synthetic_clips(smoke_cfg, tmp_path)
    assert index["source"] == "synthetic"
    assert index["credit"] is None
    assert [c["word"] for c in index["clips"]] == list(smoke_cfg.replay.words)
    assert json.loads((tmp_path / "index.json").read_text()) == index
    for clip in index["clips"]:
        sheet = Image.open(tmp_path / clip["sprite"])
        rows = -(-clip["frames"] // clip["columns"])
        assert sheet.size == (clip["columns"] * clip["frameWidth"], rows * clip["frameHeight"])
        lm = json.loads((tmp_path / clip["landmarks"]).read_text())
        n = clip["frames"]
        points = np.array(lm["points"], dtype=np.float32).reshape(n, 44, 3)
        blend = np.array(lm["blend"], dtype=np.float32).reshape(n, 52)
        matrix = np.array(lm["matrix"], dtype=np.float32).reshape(n, 4, 4)
        feats = frames_features(points, blend, matrix)
        assert feats[:, -1].all()
        a, b = clip["modelRange"]
        assert 0 <= a < b <= n
        assert b - a >= smoke_cfg.train.min_frames


def test_mouth_crop_has_the_requested_aspect() -> None:
    points = np.full((2, 44, 3), 0.5, np.float32)
    points[:, :40, 0] = np.linspace(0.4, 0.6, 40)
    points[:, :40, 1] = np.linspace(0.7, 0.75, 40)
    x, _y, w, h = mouth_crop(points, (360, 288), aspect=1.5)
    assert abs(w / h - 1.5) < 0.05
    assert x < 0.4 * 288 < 0.6 * 288 < x + w
