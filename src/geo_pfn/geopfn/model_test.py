"""Tests for the two-stage geo-PFN model and bar-distribution head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from geo_pfn.geopfn.heads import BarDistribution
from geo_pfn.geopfn.model import GeoPFN, GeoPFNConfig


def test_bar_distribution_mean_recovers_center() -> None:
    bar = BarDistribution(n_bins=64, half_range=4.0)
    # a logit spike on one bin -> mean near that bin center, small std
    logits = torch.full((1, 64), -10.0)
    logits[0, 40] = 10.0
    mean, std = bar.mean_std(logits)
    assert abs(mean.item() - bar.centers[40].item()) < 0.2
    assert std.item() < 0.3


def test_bar_distribution_loss_and_bins() -> None:
    bar = BarDistribution(n_bins=32, half_range=4.0)
    z = torch.tensor([-3.9, 0.0, 3.9, 100.0])
    bins = bar.to_bins(z)
    assert bins.min() >= 0 and bins.max() <= 31
    assert bins[0] == 0 and bins[2] == 31 and bins[3] == 31  # clamps out-of-range
    logits = torch.randn(4, 32)
    assert bar.loss(logits, z).item() > 0
    with pytest.raises(ValueError, match="n_bins"):
        BarDistribution(n_bins=1)


def make_batch(b=4, r=40, f=10, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(b, r, f)), dtype=torch.float32)
    x[torch.tensor(rng.random((b, r, f)) < 0.1)] = float("nan")
    y = torch.tensor(rng.normal(size=(b, r)), dtype=torch.float32)
    ctx = torch.tensor(rng.random((b, r)) < 0.5)
    ctx[:, 0] = True  # guarantee at least one context row
    return x, y, ctx


def test_config_validation() -> None:
    GeoPFNConfig()
    with pytest.raises(ValueError, match="divisible"):
        GeoPFNConfig(d_model=10, n_heads=3)
    with pytest.raises(ValueError, match="layers"):
        GeoPFNConfig(row_layers=0)


def test_forward_shape_and_param_count() -> None:
    torch.manual_seed(0)
    model = GeoPFN(GeoPFNConfig())
    x, y, ctx = make_batch()
    logits = model(x, y, ctx)
    assert logits.shape == (4, 40, model.config.n_bins)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1e6 < n_params < 3e7  # a few M params


def test_query_prediction_independent_of_other_query_rows() -> None:
    # a query row's output must not depend on other query rows' features
    # (keys come only from context rows), so ICL is a valid predictor.
    torch.manual_seed(0)
    model = GeoPFN(GeoPFNConfig(col_layers=1, row_layers=2)).eval()
    x, y, ctx = make_batch(b=1, r=30, f=6)
    query = (~ctx[0]).nonzero().squeeze(1)
    with torch.no_grad():
        base = model(x, y, ctx)
        x2 = x.clone()
        # perturb ONE query row's features; other query rows must be unchanged
        target = query[0].item()
        others = query[1:]
        x2[0, target] += 5.0
        pert = model(x2, y, ctx)
    torch.testing.assert_close(base[0, others], pert[0, others], atol=1e-4, rtol=1e-4)


def test_missing_column_is_safe() -> None:
    torch.manual_seed(0)
    model = GeoPFN(GeoPFNConfig()).eval()
    x, y, ctx = make_batch()
    x[:, :, 3] = float("nan")  # a fully-missing column
    with torch.no_grad():
        logits = model(x, y, ctx)
    assert torch.isfinite(logits).all()
