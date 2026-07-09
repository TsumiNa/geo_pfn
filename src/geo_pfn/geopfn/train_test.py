"""Tests for geo_pfn.geopfn.train."""

from __future__ import annotations

import pytest
import torch

from geo_pfn.geopfn.model import GeoPFN, GeoPFNConfig
from geo_pfn.geopfn.train import GeoPFNTrainConfig, _step_loss, normalize_target
from geo_pfn.geoprior.config import GeoPriorConfig


def test_train_config_validation() -> None:
    GeoPFNTrainConfig()
    with pytest.raises(ValueError, match="steps"):
        GeoPFNTrainConfig(steps=0)


def test_normalize_target_uses_context_only() -> None:
    y = torch.tensor([[1.0, 3.0, 100.0, -100.0]])
    ctx = torch.tensor([[True, True, False, False]])
    z, mean, std = normalize_target(y, ctx)
    assert abs(mean.item() - 2.0) < 1e-5  # mean of context rows (1, 3)
    assert abs(z[0, 0].item() + z[0, 1].item()) < 1e-5  # symmetric around mean


def test_step_loss_finite_and_backprops() -> None:
    torch.manual_seed(0)
    model = GeoPFN(GeoPFNConfig(col_layers=1, row_layers=2, d_model=64, n_heads=4))
    g = torch.Generator().manual_seed(0)
    loss, rmse = _step_loss(model, GeoPriorConfig(), 4, g, torch.device("cpu"))
    assert torch.isfinite(loss) and loss.item() > 0
    assert rmse >= 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(gr).all() for gr in grads)
