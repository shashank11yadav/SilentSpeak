from __future__ import annotations

import torch

from silentspeak.config import Config
from silentspeak.features import FEATURE_DIM
from silentspeak.model import LipEncoder, count_parameters, supcon_loss
from silentspeak.paths import CONFIG_DIR


def _model(name: str = "default.toml") -> LipEncoder:
    cfg = Config.load(CONFIG_DIR / name)
    torch.manual_seed(0)
    return LipEncoder(cfg.model, n_classes=45).eval()


def test_default_model_is_one_to_two_million_parameters() -> None:
    n = count_parameters(_model())
    assert 1_000_000 <= n <= 2_000_000, n


def test_outputs_have_the_contract_shapes() -> None:
    model = _model()
    logits, emb = model(torch.randn(3, 17, FEATURE_DIM))
    assert logits.shape == (3, 45)
    assert emb.shape == (3, 128)
    torch.testing.assert_close(emb.norm(dim=1), torch.ones(3))


def test_padding_does_not_change_outputs() -> None:
    model = _model()
    a = torch.randn(1, 9, FEATURE_DIM)
    b = torch.randn(1, 20, FEATURE_DIM)
    batch = torch.zeros(2, 20, FEATURE_DIM)
    batch[0, :9] = a[0]
    batch[1] = b[0]
    with torch.no_grad():
        logits, emb = model(batch, torch.tensor([9, 20]))
        la, _ea = model(a)
        lb, eb = model(b)
    torch.testing.assert_close(logits[0], la[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(emb[1], eb[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(logits[1], lb[0], atol=1e-5, rtol=1e-4)


def test_normalisation_buffers_are_applied() -> None:
    model = _model()
    x = torch.randn(1, 10, FEATURE_DIM)
    with torch.no_grad():
        before = model(x)[1]
        model.set_normalisation(torch.full((FEATURE_DIM,), 2.0), torch.full((FEATURE_DIM,), 3.0))
        after = model(x * 3 + 2)[1]
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-4)


def test_supcon_prefers_clustered_embeddings() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    centres = torch.nn.functional.normalize(torch.randn(3, 16), dim=1)
    clustered = torch.nn.functional.normalize(centres[labels] + 0.01 * torch.randn(6, 16), dim=1)
    scattered = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    good = supcon_loss(clustered, labels, temperature=0.1)
    bad = supcon_loss(scattered, labels, temperature=0.1)
    assert torch.isfinite(good)
    assert good < bad


def test_supcon_ignores_anchors_without_positives() -> None:
    emb = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    assert supcon_loss(emb, torch.tensor([0, 1, 2]), temperature=0.1) == 0
    loss = supcon_loss(emb, torch.tensor([0, 0, 2]), temperature=0.1)
    assert torch.isfinite(loss)
    assert loss > 0
