"""Tests for the mini-PFN model core properties."""

import math

import pytest
import torch

from geo_pfn.minipfn.config import ModelConfig
from geo_pfn.minipfn.model import MiniPFN, mask_class_logits


@pytest.fixture
def model() -> MiniPFN:
    torch.manual_seed(0)
    return MiniPFN(
        ModelConfig(d_model=32, n_layers=2, n_heads=2, feature_emb_dim=8)
    ).eval()


def _toy_batch(
    b: int = 3, r: int = 20, f: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(b, r, f, generator=g)
    y = torch.randint(0, 3, (b, r), generator=g)
    return x, y


def test_forward_shape_and_determinism(model: MiniPFN) -> None:
    x, y = _toy_batch()
    logits1 = model(x, y, train_size=12)
    logits2 = model(x, y, train_size=12)
    assert logits1.shape == (3, 8, model.config.max_classes)
    assert torch.isfinite(logits1).all()
    assert torch.equal(logits1, logits2)  # eval mode must be deterministic


def test_nan_inputs_produce_finite_logits(model: MiniPFN) -> None:
    x, y = _toy_batch()
    x[:, 15:, :3] = math.nan  # entire columns missing in test rows
    x[0, 2, 4] = math.nan  # plus a train cell
    logits = model(x, y, train_size=12)
    assert torch.isfinite(logits).all()


def test_test_rows_are_independent(model: MiniPFN) -> None:
    x, y = _toy_batch()
    logits_before = model(x, y, train_size=12)
    x2 = x.clone()
    x2[:, -1, :] = 100.0  # corrupt the last test row
    logits_after = model(x2, y, train_size=12)
    torch.testing.assert_close(logits_before[:, :-1], logits_after[:, :-1])


def test_test_labels_do_not_leak(model: MiniPFN) -> None:
    x, y = _toy_batch()
    y2 = y.clone()
    y2[:, 12:] = (y[:, 12:] + 1) % 3
    torch.testing.assert_close(model(x, y, train_size=12), model(x, y2, train_size=12))


def test_train_labels_matter(model: MiniPFN) -> None:
    x, y = _toy_batch()
    y2 = y.clone()
    y2[:, :12] = (y[:, :12] + 1) % 3
    assert not torch.allclose(model(x, y, train_size=12), model(x, y2, train_size=12))


def test_invalid_train_size_raises(model: MiniPFN) -> None:
    x, y = _toy_batch()
    with pytest.raises(ValueError, match="train_size"):
        model(x, y, train_size=0)
    with pytest.raises(ValueError, match="train_size"):
        model(x, y, train_size=x.shape[1])


def test_gradients_flow() -> None:
    torch.manual_seed(0)
    model = MiniPFN(
        ModelConfig(d_model=32, n_layers=2, n_heads=2, feature_emb_dim=8)
    ).train()
    x, y = _toy_batch()
    x[:, 15:, 0] = math.nan
    logits = model(x, y, train_size=12)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y[:, 12:].flatten())
    loss.backward()
    grad = model.cell_encoder.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_mask_class_logits() -> None:
    logits = torch.zeros(2, 3, 4)
    masked = mask_class_logits(logits, torch.tensor([2, 4]))
    assert bool((masked[0, :, 2:] < -1e8).all())
    assert bool((masked[0, :, :2] == 0).all())
    assert bool((masked[1] == 0).all())
