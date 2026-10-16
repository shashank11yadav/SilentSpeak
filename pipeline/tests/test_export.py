from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from silentspeak.config import Config
from silentspeak.export import check_onnx, export_onnx, export_stub, sha256
from silentspeak.features import FEATURE_DIM
from silentspeak.model import LipEncoder
from silentspeak.vocab import CATEGORIES


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory: pytest.TempPathFactory, smoke_cfg: Config) -> Path:
    out = tmp_path_factory.mktemp("models")
    export_stub(smoke_cfg, out)
    return out


def test_onnx_io_contract(stub_dir: Path) -> None:
    model = onnx.load(str(stub_dir / "model.onnx"))
    (inp,) = model.graph.input
    assert inp.name == "frames"
    dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    assert dims == [1, "T", FEATURE_DIM]
    assert [o.name for o in model.graph.output] == ["logits", "embedding"]
    assert model.opset_import[0].version == 17


@pytest.mark.parametrize("t", [5, 23, 50])
def test_onnx_runs_for_every_supported_length(stub_dir: Path, t: int) -> None:
    session = ort.InferenceSession(str(stub_dir / "model.onnx"))
    logits, emb = session.run(None, {"frames": np.zeros((1, t, FEATURE_DIM), np.float32)})
    assert logits.shape == (1, 45)
    assert emb.shape == (1, 128)
    assert abs(float(np.linalg.norm(emb)) - 1) < 1e-4


def test_stub_meta_marks_an_untrained_model(stub_dir: Path, smoke_cfg: Config) -> None:
    meta = json.loads((stub_dir / "model-meta.json").read_text())
    assert meta["metrics"] is None
    assert meta["trainedOn"] is None
    assert meta["featureDim"] == FEATURE_DIM
    assert meta["fps"] == 25
    assert meta["classes"] == list(smoke_cfg.trained_words)
    assert meta["modelSha256"] == sha256(stub_dir / "model.onnx")
    for key in (
        "version",
        "featureVersion",
        "fps",
        "featureDim",
        "classes",
        "trainedOn",
        "metrics",
    ):
        assert key in meta


def test_stub_labels_cover_every_category_without_shipping(stub_dir: Path) -> None:
    labels = json.loads((stub_dir / "labels.json").read_text())
    assert [c["id"] for c in labels["categories"]] == list(CATEGORIES)
    assert not any(c["shipped"] for c in labels["categories"])
    command = labels["categories"][0]
    assert command["words"] == ["bin", "lay", "place"]  # "set" is a few-shot word
    assert labels["fewshotWords"] == ["set", "white", "with", "k", "seven", "soon"]


def test_check_onnx_detects_a_mismatch(tmp_path: Path, smoke_cfg: Config) -> None:
    torch.manual_seed(0)
    model = LipEncoder(smoke_cfg.model, 45).eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path, 50)
    with torch.no_grad():
        model.inp.weight.add_(0.5)
    with pytest.raises(RuntimeError, match="differs"):
        check_onnx(model, path, 50)
