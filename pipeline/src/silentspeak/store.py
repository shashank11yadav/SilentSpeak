"""On-disk format of extracted landmarks: one ``sN.npz`` per speaker.

Arrays (``N`` = total frames of the speaker, ``U`` = utterances):

* ``utterances`` ``str[U]`` and ``offsets`` ``int64[U + 1]``: utterance ``i`` is
  frames ``offsets[i]:offsets[i + 1]``.
* ``points`` ``float32[N, 44, 3]``: the stored face-mesh points in image-height
  units (see ``features.py``); NaN rows where no face was found.
* ``blend`` ``float32[N, 52]`` blendshape scores, ``matrix`` ``float32[N, 4, 4]``.
* ``frame_size`` ``int64[2]`` (width, height) and ``format_version``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import numpy as np
from numpy.typing import NDArray

from .landmarks import N_BLENDSHAPES, N_STORED

FORMAT_VERSION: Final[int] = 1


@dataclass
class SpeakerLandmarks:
    utterances: list[str]
    offsets: NDArray[np.int64]
    points: NDArray[np.float32]
    blend: NDArray[np.float32]
    matrix: NDArray[np.float32]
    frame_size: tuple[int, int]

    def utterance(
        self, i: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.points[a:b], self.blend[a:b], self.matrix[a:b]

    @classmethod
    def from_utterances(
        cls,
        items: list[tuple[str, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]],
        frame_size: tuple[int, int],
    ) -> Self:
        lengths = [len(p) for _, p, _, _ in items]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        empty_p = np.zeros((0, N_STORED, 3), np.float32)
        empty_b = np.zeros((0, N_BLENDSHAPES), np.float32)
        empty_m = np.zeros((0, 4, 4), np.float32)
        return cls(
            utterances=[u for u, _, _, _ in items],
            offsets=offsets,
            points=np.concatenate([empty_p, *(p for _, p, _, _ in items)]).astype(np.float32),
            blend=np.concatenate([empty_b, *(b for _, _, b, _ in items)]).astype(np.float32),
            matrix=np.concatenate([empty_m, *(m for _, _, _, m in items)]).astype(np.float32),
            frame_size=frame_size,
        )

    def save(self, path: Path) -> None:
        """Write atomically, so an interrupted run never leaves a half file behind."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial.npz")
        np.savez_compressed(
            tmp,
            utterances=np.array(self.utterances, dtype=str),
            offsets=self.offsets,
            points=self.points,
            blend=self.blend,
            matrix=self.matrix,
            frame_size=np.array(self.frame_size, dtype=np.int64),
            format_version=np.array(FORMAT_VERSION),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path, allow_pickle=False) as z:
            version = int(z["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"{path}: format {version}, expected {FORMAT_VERSION}")
            w, h = (int(v) for v in z["frame_size"])
            return cls(
                utterances=[str(u) for u in z["utterances"]],
                offsets=z["offsets"].astype(np.int64),
                points=z["points"].astype(np.float32),
                blend=z["blend"].astype(np.float32),
                matrix=z["matrix"].astype(np.float32),
                frame_size=(w, h),
            )
