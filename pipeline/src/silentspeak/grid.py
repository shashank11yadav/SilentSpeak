"""Readers for the GRID corpus as distributed on Zenodo (record 3625687).

Layout, checked against the real files:

* ``alignments.zip`` holds ``alignments/sN/<utterance>.align`` for all 34
  speakers, 1000 utterances each (plus macOS ``__MACOSX/`` junk). 19 of the
  folders carry the wrong speaker label; ``read_alignments`` matches folders
  to videos by utterance id. Each line is ``start end token``; times are in
  ticks of 1/25 000 s, so a 3 s utterance ends at 74 500 and one video frame
  (25 fps) is 1000 ticks. ``sil`` and ``sp``
  mark silence and short pauses.
* ``sN.zip`` holds the speaker's 1000 videos (``sN/<utterance>.mpg``); see ``video.py``.
* ``manifest.tsv`` next to ``raw/`` lists ``key, md5, size, url`` for every
  file. A zip is only read once its size matches, so a download in progress is
  never opened.
"""

from __future__ import annotations

import math
import zipfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .vocab import NON_WORDS

TICKS_PER_FRAME: Final[float] = 1000.0
GRID_FPS: Final[float] = 25.0
SPEAKERS_WITHOUT_VIDEO: Final[frozenset[str]] = frozenset({"s21"})


@dataclass(frozen=True)
class Word:
    word: str
    start: float  # in video frames
    end: float


def parse_alignment(text: str) -> list[Word]:
    """Words of one ``.align`` file, without silences and pauses."""
    words: list[Word] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ValueError(f"line {lineno}: expected 'start end token', got {line!r}")
        start, end, token = int(parts[0]), int(parts[1]), parts[2]
        if token in NON_WORDS:
            continue
        words.append(Word(token, start / TICKS_PER_FRAME, end / TICKS_PER_FRAME))
    return words


def word_window(word: Word, pad: int, n_frames: int) -> tuple[int, int]:
    """Frame range ``[start, stop)`` covering the word plus ``pad`` frames each side."""
    start = max(0, math.floor(word.start) - pad)
    stop = min(n_frames, math.ceil(word.end) + pad)
    return start, stop


MIN_ALIGNMENT_COVERAGE: Final[float] = 0.9


def _alignment_names(zf: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    """``folder -> {utterance id -> member name}`` for every ``alignments/sN/*.align``."""
    folders: dict[str, dict[str, str]] = {}
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) == 3 and parts[0] == "alignments" and parts[2].endswith(".align"):
            folders.setdefault(parts[1], {})[parts[2][: -len(".align")]] = name
    return folders


def read_alignments(
    path: Path, speaker: str, utterances: Collection[str] | None = None
) -> dict[str, list[Word]]:
    """All alignments of one speaker, keyed by utterance id.

    The Zenodo archive labels 19 speakers' alignment folders with another
    speaker's number (s5 and s6 are swapped, s20's are filed as s21, ...).
    Utterance ids are unique to a speaker, so when ``utterances`` (the ids of
    the speaker's videos) is given, the folder that covers them is used,
    preferring the speaker's own label on a tie.
    """
    with zipfile.ZipFile(path) as zf:
        folders = _alignment_names(zf)
        folder = speaker
        if utterances is not None:
            wanted = set(utterances)

            def coverage(name: str) -> int:
                return len(wanted & folders[name].keys())

            folder = max(folders, key=lambda name: (coverage(name), name == speaker))
            if coverage(folder) < MIN_ALIGNMENT_COVERAGE * len(wanted):
                raise LookupError(
                    f"no alignment folder covers {speaker}'s videos "
                    f"(best: {folder}, {coverage(folder)}/{len(wanted)})"
                )
        names = folders.get(folder, {})
        return {utt: parse_alignment(zf.read(name).decode("ascii")) for utt, name in names.items()}


def read_manifest(path: Path) -> dict[str, int]:
    """``file name -> expected size`` from ``manifest.tsv``."""
    sizes: dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].isdigit():
            sizes[parts[0]] = int(parts[2])
    return sizes


def complete_zips(raw: Path) -> set[str]:
    """Names of zips in ``raw`` whose size matches the manifest (fully downloaded)."""
    manifest = raw.parent / "manifest.tsv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; it lists the expected file sizes")
    done: set[str] = set()
    for name, size in read_manifest(manifest).items():
        path = raw / name
        if path.is_file() and path.stat().st_size == size:
            done.add(name)
    return done


def speaker_sort_key(speaker: str) -> int:
    return int(speaker.removeprefix("s"))


def video_speakers(raw: Path) -> list[str]:
    """Speakers whose video zip is fully downloaded, in numeric order."""
    names = complete_zips(raw)
    speakers = [
        n.removesuffix(".zip")
        for n in names
        if n.startswith("s") and n.removesuffix(".zip")[1:].isdigit()
    ]
    return sorted((s for s in speakers if s not in SPEAKERS_WITHOUT_VIDEO), key=speaker_sort_key)
