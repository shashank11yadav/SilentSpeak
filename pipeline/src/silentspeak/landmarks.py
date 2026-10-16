"""Which face-mesh points SilentSpeak stores, and how they pair up under a mirror.

Indices refer to MediaPipe's 478-point face mesh. They are the same in the
Python and web landmarkers, and ``web/src/ml/landmarks.ts`` repeats them.
"""

from __future__ import annotations

from typing import Final

# The 40 points on the inner and outer lip contours, sorted.
LIP_INDICES: Final[tuple[int, ...]] = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
    185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375,
    402, 405, 409, 415,
)  # fmt: skip

# Outer eye corners (scale reference), nose tip and chin (mouth-region crop).
EYE_OUTER: Final[tuple[int, int]] = (33, 263)
NOSE_TIP: Final[int] = 1
CHIN: Final[int] = 152

STORED_INDICES: Final[tuple[int, ...]] = (*LIP_INDICES, *EYE_OUTER, NOSE_TIP, CHIN)
N_STORED: Final[int] = len(STORED_INDICES)
N_LIPS: Final[int] = len(LIP_INDICES)

# Lip points on the face's vertical midline map onto themselves under a mirror.
LIP_MIDLINE: Final[tuple[int, ...]] = (0, 13, 14, 17)

# Each pair swaps places when the image is mirrored (person's right <-> left).
LIP_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (61, 291),  # outer corners
    (78, 308),  # inner corners
    (185, 409), (40, 270), (39, 269), (37, 267),  # upper outer
    (146, 375), (91, 321), (181, 405), (84, 314),  # lower outer
    (191, 415), (80, 310), (81, 311), (82, 312),  # upper inner
    (95, 324), (88, 318), (178, 402), (87, 317),  # lower inner
)  # fmt: skip

# Upper and lower inner-lip midpoints, for the mouth-opening signal.
INNER_UPPER: Final[int] = 13
INNER_LOWER: Final[int] = 14

# MediaPipe's 52 face blendshapes, in the order the landmarker returns them.
BLENDSHAPE_NAMES: Final[tuple[str, ...]] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)  # fmt: skip
N_BLENDSHAPES: Final[int] = len(BLENDSHAPE_NAMES)

# Mouth blendshapes that have no left/right twin.
BLEND_SINGLE: Final[tuple[str, ...]] = (
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
)  # fmt: skip

# Left/right twins; the features use their sum, which a mirror leaves unchanged.
BLEND_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("mouthLeft", "mouthRight"),
    ("mouthSmileLeft", "mouthSmileRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthDimpleLeft", "mouthDimpleRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthPressLeft", "mouthPressRight"),
    ("mouthLowerDownLeft", "mouthLowerDownRight"),
    ("mouthUpperUpLeft", "mouthUpperUpRight"),
)


def stored_position(mesh_index: int) -> int:
    """Row of a face-mesh index in the stored ``[N_STORED, 3]`` array."""
    return STORED_INDICES.index(mesh_index)


def mirror_permutation() -> list[int]:
    """Stored-row permutation that swaps each point with its mirror twin."""
    perm = list(range(N_STORED))
    for a, b in LIP_PAIRS:
        ia, ib = stored_position(a), stored_position(b)
        perm[ia], perm[ib] = ib, ia
    ia, ib = stored_position(EYE_OUTER[0]), stored_position(EYE_OUTER[1])
    perm[ia], perm[ib] = ib, ia
    return perm


def blend_mirror_permutation() -> list[int]:
    """Blendshape permutation that swaps every ...Left with its ...Right."""
    perm = list(range(N_BLENDSHAPES))
    for i, name in enumerate(BLENDSHAPE_NAMES):
        if name.endswith("Left"):
            j = BLENDSHAPE_NAMES.index(name[: -len("Left")] + "Right")
            perm[i], perm[j] = j, i
    return perm
