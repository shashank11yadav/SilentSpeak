"""ONNX export, the web app's model files and the shared feature-parity fixture.

The ONNX I/O contract (also in ``model-meta.json``):

* input ``frames``: ``float32[1, T, 157]``, T from 5 to 50, 25 fps sequence
  features (``features.sequence_features``);
* output ``logits``: ``float32[1, 45]``, scaled cosine scores for the trained words
  in ``model-meta.json`` ``classes`` order;
* output ``embedding``: ``float32[1, 128]``, L2-normalised.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray

from .config import Config
from .dataset import FEATURE_VERSION
from .features import (
    FEATURE_DIM,
    FPS,
    FRAME_DIM,
    frame_features,
    frames_features,
    mirror_frame,
    resample,
    sequence_features,
)
from .landmarks import BLENDSHAPE_NAMES, STORED_INDICES
from .model import LipEncoder
from .synthetic import make_speaker, render_frames, word_states
from .vocab import CATEGORIES

log = logging.getLogger(__name__)

OPSET = 17
PARITY_TOLERANCE = 1e-4
STUB_THRESHOLDS = {"category": 0.5, "protoMargin": 0.03, "protoMinSimilarity": 0.0}


class _OnnxWrapper(torch.nn.Module):
    def __init__(self, model: LipEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(frames)  # type: ignore[no-any-return]


def export_onnx(model: LipEncoder, path: Path, max_frames: int) -> None:
    """Write the model as a single-file ONNX graph and check it against PyTorch."""
    model = model.cpu().eval()
    wrapper = _OnnxWrapper(model).eval()
    example = torch.zeros(1, 20, FEATURE_DIM)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The TorchScript-based exporter keeps the GRU as one ONNX GRU node and
        # the time axis fully dynamic; the dynamo exporter specialises it.
        torch.onnx.export(
            wrapper,
            (example,),
            str(path),
            input_names=["frames"],
            output_names=["logits", "embedding"],
            dynamic_axes={"frames": {1: "T"}},
            opset_version=OPSET,
            dynamo=False,
        )
    model.eval()
    check_onnx(model, path, max_frames)


def check_onnx(model: LipEncoder, path: Path, max_frames: int) -> float:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for t in (5, 17, max_frames):
        x = torch.randn(1, t, FEATURE_DIM, generator=gen)
        with torch.no_grad():
            ref = [o.numpy() for o in model(x)]
        got = session.run(None, {"frames": x.numpy()})
        for a, b in zip(ref, got, strict=True):
            worst = max(worst, float(np.abs(a - b).max()))
    if worst > PARITY_TOLERANCE:
        raise RuntimeError(f"ONNX output differs from PyTorch by {worst:.2e}")
    return worst


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(x: Any) -> Any:
    """JSON-safe copy: NaN becomes null."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def write_web_files(
    out_dir: Path,
    cfg: Config,
    words: list[str],
    metrics: dict[str, Any] | None,
    model_path: Path,
) -> None:
    """``model-meta.json`` and ``labels.json`` for the web app."""
    by_cat = {c["id"]: c for c in metrics["categories"]} if metrics else {}
    categories = []
    for cat, all_words in CATEGORIES.items():
        m = by_cat.get(cat)
        categories.append(
            {
                "id": cat,
                "words": [w for w in all_words if w in words],
                "shipped": bool(m["shipped"]) if m else False,
                "top1": m["top1"] if m else None,
                "top3": m["top3"] if m else None,
                "chance1": 1 / sum(w in words for w in all_words),
                "threshold": m["threshold"] if m else STUB_THRESHOLDS["category"],
            }
        )
    labels = {
        "classes": words,
        "categories": categories,
        "fewshotWords": list(cfg.data.fewshot_words),
    }
    summary = None
    trained_on = None
    thresholds = {
        "protoMargin": STUB_THRESHOLDS["protoMargin"],
        "protoMinSimilarity": STUB_THRESHOLDS["protoMinSimilarity"],
    }
    if metrics:
        sp = metrics["speakers"]
        trained_on = (
            f"GRID corpus (Cooke et al., CC BY 4.0): {len(sp['train'])} training speakers, "
            f"{len(sp['val'])} validation speakers; tested on held-out speakers "
            f"{', '.join(sp['test'])}"
        )
        summary = {
            "overallTop1": metrics["overall"]["top1"],
            "fewshotAccuracy": metrics["fewshot"]["accuracy"],
            "fewshotWords": metrics["fewshot"]["words"],
            "testSpeakers": sp["test"],
        }
        thresholds = {
            "protoMargin": metrics["fewshot"]["margin"],
            "protoMinSimilarity": metrics["fewshot"]["minSimilarity"],
        }
    meta = {
        "version": 1,
        "featureVersion": FEATURE_VERSION,
        "fps": FPS,
        "frameDim": FRAME_DIM,
        "featureDim": FEATURE_DIM,
        "minFrames": cfg.train.min_frames,
        "maxFrames": cfg.train.max_frames,
        "embeddingDim": cfg.model.embedding_dim,
        "input": "frames",
        "outputs": ["logits", "embedding"],
        "classes": words,
        "trainedOn": trained_on,
        "metrics": summary,
        "thresholds": thresholds,
        "modelSha256": sha256(model_path),
    }
    (out_dir / "labels.json").write_text(json.dumps(_clean(labels), indent=2) + "\n")
    (out_dir / "model-meta.json").write_text(json.dumps(_clean(meta), indent=2) + "\n")


def export_stub(cfg: Config, out_dir: Path) -> Path:
    """A random-weight model with the real I/O contract, so the app runs before training."""
    torch.manual_seed(cfg.seed)
    words = list(cfg.trained_words)
    model = LipEncoder(cfg.model, n_classes=len(words)).eval()
    path = out_dir / "model.onnx"
    export_onnx(model, path, cfg.train.max_frames)
    write_web_files(out_dir, cfg, words, None, path)
    return path


def export_trained(
    cfg: Config, model: LipEncoder, words: list[str], metrics: dict[str, Any] | None, out_dir: Path
) -> Path:
    path = out_dir / "model.onnx"
    export_onnx(model, path, cfg.train.max_frames)
    write_web_files(out_dir, cfg, words, metrics, path)
    return path


# ---------------------------------------------------------------------------
# Feature-parity fixture


def _nested(a: NDArray[Any]) -> Any:
    return _clean(np.asarray(a, dtype=np.float64).tolist())


def build_fixture(seed: int = 11) -> dict[str, Any]:
    """Inputs and expected outputs that the Python and TypeScript tests both check."""
    rng = np.random.default_rng(seed)
    speaker = make_speaker(rng)
    states = word_states("please", 24, 1.0)
    points, blend, matrix = render_frames(states, speaker, rng)
    points[7] = np.nan  # a frame where the face was lost
    blend[7] = np.nan
    matrix[7] = np.nan
    aspect = 360 / 288

    frames = []
    for name, idx in (("frontal", 3), ("open", 12), ("missing", 7)):
        frames.append(
            {
                "name": name,
                "points": _nested(points[idx]),
                "blend": _nested(blend[idx]),
                "matrix": _nested(matrix[idx]),
                "expected": _nested(frame_features(points[idx], blend[idx], matrix[idx])),
            }
        )
    mp, mb, mm = mirror_frame(points[12], blend[12], matrix[12], aspect)
    frames.append(
        {
            "name": "open-mirrored",
            "points": _nested(mp),
            "blend": _nested(mb),
            "matrix": _nested(mm),
            "expected": _nested(frame_features(points[12], blend[12], matrix[12])),
        }
    )

    # The full live path: per-frame features at 30 fps -> 25 fps -> deltas.
    feats = frames_features(points, blend, matrix)
    expected_seq = sequence_features(resample(feats, 30.0, FPS))
    mirrored = [
        mirror_frame(p, b, m, aspect) for p, b, m in zip(points, blend, matrix, strict=True)
    ]
    return {
        "featureVersion": FEATURE_VERSION,
        "frameDim": FRAME_DIM,
        "featureDim": FEATURE_DIM,
        "storedIndices": list(STORED_INDICES),
        "blendshapeNames": list(BLENDSHAPE_NAMES),
        "tolerance": PARITY_TOLERANCE,
        "frames": frames,
        "sequence": {
            "srcFps": 30.0,
            "dstFps": FPS,
            "points": _nested(points),
            "blend": _nested(blend),
            "matrix": _nested(matrix),
            "mirroredPoints": _nested(np.stack([m[0] for m in mirrored])),
            "mirroredBlend": _nested(np.stack([m[1] for m in mirrored])),
            "mirroredMatrix": _nested(np.stack([m[2] for m in mirrored])),
            "frameFeatures": _nested(feats),
            "resampled": _nested(resample(feats, 30.0, FPS)),
            "expected": _nested(expected_seq),
        },
    }


def write_fixture(path: Path, seed: int = 11) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_fixture(seed)) + "\n")
