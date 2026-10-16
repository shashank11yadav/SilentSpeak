"""Readers for the GRID corpus as distributed on Zenodo (record 3625687).

Layout, checked against the real files:

* ``alignments.zip`` holds ``alignments/sN/<utterance>.align`` for all 34
  speakers, 1000 utterances each (plus macOS ``__MACOSX/`` junk). Each line is
  ``start end token``; times are in ticks of 1/25 000 s, so a 3 s utterance
  ends at 74 500 and one video frame (25 fps) is 1000 ticks. ``sil`` and ``sp``
  mark silence and short pauses.
* ``sN.zip`` holds the speaker's 1000 videos (``sN/<utterance>.mpg``); see ``video.py``.
* ``manifest.tsv`` next to ``raw/`` lists ``key, md5, size, url`` for every
  file. A zip is only read once its size matches, so a download in progress is
  never opened.
"""

from __future__ import annotations

import math
import zipfile
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


def read_alignments(path: Path, speaker: str) -> dict[str, list[Word]]:
    """All alignments of one speaker, keyed by utterance id."""
    prefix = f"alignments/{speaker}/"
    out: dict[str, list[Word]] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(".align"):
                utt = name[len(prefix) : -len(".align")]
                if "/" in utt:
                    continue
                out[utt] = parse_alignment(zf.read(name).decode("ascii"))
    return out


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
