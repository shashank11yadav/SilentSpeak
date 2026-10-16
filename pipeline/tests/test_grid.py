from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from silentspeak.grid import (
    Word,
    complete_zips,
    parse_alignment,
    read_alignments,
    speaker_sort_key,
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
