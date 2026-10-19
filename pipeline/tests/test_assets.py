from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

from silentspeak import assets

BODY = b"face landmarker bytes"


def _serve(monkeypatch: pytest.MonkeyPatch, body: bytes) -> list[str]:
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> Any:
        calls.append(url)
        return io.BytesIO(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(assets, "FACE_LANDMARKER_SHA256", hashlib.sha256(BODY).hexdigest())
    return calls


def test_download_is_written_only_after_its_hash_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, BODY)
    path = assets.face_landmarker_model(tmp_path)
    assert path.read_bytes() == BODY
    assert sorted(p.name for p in tmp_path.iterdir()) == ["face_landmarker.task"]


def test_a_tampered_download_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        assets.face_landmarker_model(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_oversized_download_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, BODY)
    monkeypatch.setattr(assets, "MAX_MODEL_BYTES", 4)
    with pytest.raises(RuntimeError, match="larger than"):
        assets.face_landmarker_model(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_verified_copy_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _serve(monkeypatch, BODY)
    (tmp_path / "face_landmarker.task").write_bytes(BODY)
    assets.face_landmarker_model(tmp_path)
    assert calls == []
