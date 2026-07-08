"""Tests for geo_pfn.geoprior.train."""

from __future__ import annotations

import pytest
import torch

from geo_pfn.geoprior.config import GeoPriorConfig
from geo_pfn.geoprior.train import GeoTrainConfig, geo_task


def test_train_config_validation() -> None:
    GeoTrainConfig()
    with pytest.raises(ValueError, match="n_bins"):
        GeoTrainConfig(n_bins=5)
    with pytest.raises(ValueError, match="steps"):
        GeoTrainConfig(steps=0)


def test_geo_task_shapes_and_label_range() -> None:
    g = torch.Generator().manual_seed(0)
    x, y, train_size = geo_task(GeoPriorConfig(), 8, 4, (0.3, 0.8), g)
    b, r, _ = x.shape
    assert b == 8 and y.shape == (8, r)
    assert 0 < train_size < r
    assert y.min() >= 0 and y.max() <= 3
    assert y.dtype == torch.long


def test_geo_task_bins_use_full_range_pooled() -> None:
    # A single layer-dominated borehole can have collapsed bins, but pooled over
    # the batch every quantile bin is populated (edges track the context spread).
    g = torch.Generator().manual_seed(1)
    _, y, train_size = geo_task(GeoPriorConfig(), 16, 4, (0.5, 0.5), g)
    ctx = y[:, :train_size]
    assert (torch.bincount(ctx.flatten(), minlength=4) > 0).all()


def test_geo_task_labels_reproducible() -> None:
    x1, y1, t1 = geo_task(
        GeoPriorConfig(), 4, 4, (0.4, 0.4), torch.Generator().manual_seed(7)
    )
    x2, y2, t2 = geo_task(
        GeoPriorConfig(), 4, 4, (0.4, 0.4), torch.Generator().manual_seed(7)
    )
    assert t1 == t2
    torch.testing.assert_close(x1, x2, equal_nan=True)  # block-missing cells are NaN
    torch.testing.assert_close(y1, y2)
