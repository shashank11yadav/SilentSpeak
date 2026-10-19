from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from silentspeak import video as video_module
from silentspeak.extract import extract_all
from silentspeak.store import SpeakerLandmarks
from silentspeak.video import SpeakerVideo


def _mpg(path: Path, frames: int, colour: int) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"PIM1"), 25, (360, 288))
    assert writer.isOpened()
    for i in range(frames):
        img = np.full((288, 360, 3), colour, np.uint8)
        cv2.circle(img, (180, 144), 20 + i, (40, 40, 200), -1)
        writer.write(img)
    writer.release()
    return path.read_bytes()


def grid_like(root: Path, speaker: str, utterances: int, frames: int) -> Path:
    """A speaker zip with the real Zenodo layout (videos, Thumbs.db, __MACOSX)."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zpath = raw / f"{speaker}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{speaker}/", "")
        zf.writestr(f"{speaker}/Thumbs.db", b"\0" * 10)
        for u in range(utterances):
            data = _mpg(root / f"u{u}.mpg", frames, 120 + 10 * u)
            zf.writestr(f"{speaker}/utt{u:03d}.mpg", data)
            zf.writestr(f"__MACOSX/{speaker}/._utt{u:03d}.mpg", b"junk")
    return zpath


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else "0" * 32  # noqa: S324


def _manifest(root: Path, sizes: dict[str, int]) -> None:
    (root / "manifest.tsv").write_text(
        "".join(
            f"{k}\tmd5:{_md5(root / 'raw' / k)}\t{v}\thttps://example.invalid/{k}\n"
            for k, v in sizes.items()
        )
    )


def test_speaker_video_reads_the_zenodo_layout(tmp_path: Path) -> None:
    zpath = grid_like(tmp_path, "s7", utterances=2, frames=6)
    with SpeakerVideo(zpath) as video:
        assert video.utterances == ["utt000", "utt001"]
        frames = list(video.frames("utt001"))
    assert len(frames) == 6
    assert frames[0].shape == (288, 360, 3)
    assert frames[0].dtype == np.uint8


def test_speaker_video_refuses_an_oversized_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zpath = grid_like(tmp_path, "s7", utterances=1, frames=3)
    monkeypatch.setattr(video_module, "MAX_VIDEO_BYTES", 16)
    with SpeakerVideo(zpath) as video, pytest.raises(ValueError, match="too large"):
        next(video.frames("utt000"))


def test_extract_refuses_a_zip_that_does_not_match_its_md5(tmp_path: Path) -> None:
    zpath = grid_like(tmp_path, "s7", utterances=1, frames=3)
    size = zpath.stat().st_size
    (tmp_path / "manifest.tsv").write_text(
        f"s7.zip\tmd5:{'0' * 32}\t{size}\thttps://example.invalid/s7.zip\n"
    )
    with pytest.raises(ValueError, match=r"s7\.zip.*md5"):
        extract_all(tmp_path, tmp_path / "landmarks", workers=1)
    assert not (tmp_path / "landmarks" / "s7.npz").exists()


@pytest.mark.mediapipe
def test_extract_writes_one_file_per_speaker_and_resumes(tmp_path: Path) -> None:
    zpath = grid_like(tmp_path, "s7", utterances=2, frames=5)
    grid_like(tmp_path, "s8", utterances=1, frames=5)
    # s8 is still downloading: its size does not match the manifest yet.
    _manifest(tmp_path, {"s7.zip": zpath.stat().st_size, "s8.zip": 10**9})
    out = tmp_path / "landmarks"

    stats = extract_all(tmp_path, out, workers=1)
    assert stats["not_ready"] == []
    assert set(stats["speakers"]) == {"s7"}
    lm = SpeakerLandmarks.load(out / "s7.npz")
    assert lm.utterances == ["utt000", "utt001"]
    assert lm.offsets.tolist() == [0, 5, 10]
    assert lm.frame_size == (360, 288)
    # No face in a synthetic frame: every row is NaN, never a made-up face.
    assert np.isnan(lm.points).all()
    assert stats["speakers"]["s7"]["missing"] == 10
    assert not (out / "s8.npz").exists()

    again = extract_all(tmp_path, out, workers=1)
    assert again["skipped"] == ["s7"]
    assert again["speakers"] == {}


@pytest.mark.mediapipe
def test_extract_limits_utterances_and_reports_unready_speakers(tmp_path: Path) -> None:
    zpath = grid_like(tmp_path, "s7", utterances=3, frames=5)
    _manifest(tmp_path, {"s7.zip": zpath.stat().st_size})
    stats = extract_all(tmp_path, tmp_path / "lm", speakers=["s7", "s9"], limit_utterances=1)
    assert stats["not_ready"] == ["s9"]
    assert SpeakerLandmarks.load(tmp_path / "lm" / "s7.npz").utterances == ["utt000"]
