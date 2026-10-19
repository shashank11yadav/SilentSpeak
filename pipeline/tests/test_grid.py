from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from silentspeak import grid
from silentspeak.grid import (
    Word,
    complete_zips,
    parse_alignment,
    read_alignments,
    speaker_sort_key,
    verify_download,
    word_window,
)

ALIGN = """0 23750 sil
23750 29500 bin
29500 34000 blue
34000 35500 at
35500 41000 f
41000 47250 two
47250 53000 now
53000 74500 sil
"""


def test_parse_alignment_converts_25khz_ticks_to_frames() -> None:
    words = parse_alignment(ALIGN)
    assert [w.word for w in words] == ["bin", "blue", "at", "f", "two", "now"]
    assert words[0] == Word("bin", 23.75, 29.5)
    assert words[-1] == Word("now", 47.25, 53.0)


def test_parse_alignment_drops_short_pauses_and_blank_lines() -> None:
    words = parse_alignment("0 1000 sil\n1000 2000 sp\n\n2000 9000 red\n")
    assert words == [Word("red", 2.0, 9.0)]


def test_parse_alignment_rejects_malformed_lines() -> None:
    with pytest.raises(ValueError, match="line 1"):
        parse_alignment("0 sil\n")


def test_word_window_pads_and_clips() -> None:
    assert word_window(Word("bin", 23.75, 29.5), pad=2, n_frames=75) == (21, 32)
    assert word_window(Word("bin", 0.5, 3.0), pad=2, n_frames=75) == (0, 5)
    assert word_window(Word("bin", 70.0, 74.5), pad=2, n_frames=75) == (68, 75)


def test_read_alignments_uses_real_zip_layout(tmp_path: Path) -> None:
    path = tmp_path / "alignments.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("alignments/", "")
        zf.writestr("alignments/s1/bbaf2n.align", ALIGN)
        zf.writestr("alignments/s1/bbaf3s.align", ALIGN)
        zf.writestr("alignments/s2/bbaf2n.align", ALIGN)
        zf.writestr("__MACOSX/alignments/s1/._bbaf2n.align", "junk")
    got = read_alignments(path, "s1")
    assert sorted(got) == ["bbaf2n", "bbaf3s"]
    assert got["bbaf2n"][0].word == "bin"


def _swapped_alignments(tmp_path: Path) -> Path:
    # On Zenodo, 19 speakers' alignment folders carry another speaker's label
    # (s5 <-> s6, s10 -> s13, s20 -> s21, ...); utterance ids are unique per speaker.
    path = tmp_path / "alignments.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for folder, utts in (("s5", ["pwij3p", "lgwk8a"]), ("s6", ["bbae8n", "sria5s"])):
            for utt in utts:
                zf.writestr(f"alignments/{folder}/{utt}.align", ALIGN)
    return path


def test_read_alignments_finds_a_mislabelled_folder_by_utterance_ids(tmp_path: Path) -> None:
    got = read_alignments(_swapped_alignments(tmp_path), "s5", ["bbae8n", "sria5s"])
    assert sorted(got) == ["bbae8n", "sria5s"]


def test_read_alignments_prefers_the_speakers_own_folder(tmp_path: Path) -> None:
    got = read_alignments(_swapped_alignments(tmp_path), "s6", ["bbae8n", "sria5s"])
    assert sorted(got) == ["bbae8n", "sria5s"]


def test_read_alignments_refuses_when_no_folder_matches_the_videos(tmp_path: Path) -> None:
    with pytest.raises(LookupError, match="s7"):
        read_alignments(_swapped_alignments(tmp_path), "s7", ["zzzz1a", "zzzz2a"])


def test_complete_zips_checks_sizes_against_the_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "s1.zip").write_bytes(b"x" * 10)
    (raw / "s2.zip").write_bytes(b"x" * 4)  # still downloading
    (raw / "alignments.zip").write_bytes(b"x" * 3)
    (tmp_path / "manifest.tsv").write_text(
        "s1.zip\tmd5:0\t10\thttps://e/s1\ns2.zip\tmd5:0\t9\thttps://e/s2\n"
        "s3.zip\tmd5:0\t9\thttps://e/s3\nalignments.zip\tmd5:0\t3\thttps://e/a\n"
    )
    assert complete_zips(raw) == {"s1.zip", "alignments.zip"}


def test_complete_zips_requires_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError, match=r"manifest\.tsv"):
        complete_zips(tmp_path / "raw")


def test_speaker_sort_key_is_numeric() -> None:
    assert sorted(["s10", "s2", "s1"], key=speaker_sort_key) == ["s1", "s2", "s10"]


def _raw_with_manifest(tmp_path: Path, body: bytes, md5: str) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "s1.zip").write_bytes(body)
    (tmp_path / "manifest.tsv").write_text(f"s1.zip\tmd5:{md5}\t{len(body)}\thttps://e/s1\n")
    return raw


def test_verify_download_accepts_the_manifest_md5(tmp_path: Path) -> None:
    body = b"zip bytes"
    raw = _raw_with_manifest(tmp_path, body, hashlib.md5(body).hexdigest())  # noqa: S324
    verify_download(raw, "s1.zip")


def test_verify_download_rejects_a_different_file(tmp_path: Path) -> None:
    raw = _raw_with_manifest(tmp_path, b"zip bytes", hashlib.md5(b"other").hexdigest())  # noqa: S324
    with pytest.raises(ValueError, match=r"s1\.zip.*md5"):
        verify_download(raw, "s1.zip")


@pytest.mark.parametrize("md5", ["x", "synthetic", "0" * 31, ""])
def test_verify_download_needs_a_real_md5_in_the_manifest(tmp_path: Path, md5: str) -> None:
    raw = _raw_with_manifest(tmp_path, b"zip bytes", md5)
    with pytest.raises(ValueError, match="no md5"):
        verify_download(raw, "s1.zip")


def test_read_alignments_refuses_an_oversized_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _swapped_alignments(tmp_path)
    monkeypatch.setattr(grid, "MAX_ALIGNMENT_BYTES", 8)
    with pytest.raises(ValueError, match="too large"):
        read_alignments(path, "s5")
