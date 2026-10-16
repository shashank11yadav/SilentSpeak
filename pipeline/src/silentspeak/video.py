"""Frame access for GRID speaker zips, decoded in memory without unpacking.

Layout of ``sN.zip`` on Zenodo (checked against s1): ``sN/<utterance>.mpg``
for 1000 utterances, MPEG-1 video at 360x288 and 25 fps, 75 frames each,
deflate-compressed in the zip. The zip also carries ``sN/Thumbs.db`` and
macOS ``__MACOSX/`` entries, which are ignored.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from numpy.typing import NDArray

U8 = NDArray[np.uint8]
VIDEO_SUFFIX = ".mpg"


class SpeakerVideo:
    """Utterances of one speaker zip and their RGB frames."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip = zipfile.ZipFile(path)
        self._names: dict[str, str] = {}
        for info in self._zip.infolist():
            name = PurePosixPath(info.filename)
            if info.is_dir() or "__MACOSX" in name.parts or name.suffix != VIDEO_SUFFIX:
                continue
            self._names[name.stem] = info.filename

    def __enter__(self) -> SpeakerVideo:
        return self

    def __exit__(self, *exc: object) -> None:
        self._zip.close()

    @property
    def utterances(self) -> list[str]:
        return sorted(self._names)

    def frames(self, utterance: str) -> Iterator[U8]:
        """RGB ``uint8[H, W, 3]`` frames in display order."""
        # OpenCV's FFmpeg backend reads straight from a file-like object.
        data = io.BytesIO(self._zip.read(self._names[utterance]))
        cap = cv2.VideoCapture(data, cv2.CAP_FFMPEG, [])  # type: ignore[call-overload]
        if not cap.isOpened():
            raise ValueError(f"{self.path}: cannot decode {utterance}")
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        finally:
            cap.release()
