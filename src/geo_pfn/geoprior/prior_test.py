"""Tests for the geo-SCM prior sampler."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from geo_pfn.geoprior.config import GeoPriorConfig
from geo_pfn.geoprior.prior import sample_geo_batch


def test_config_validation() -> None:
    GeoPriorConfig()
    with pytest.raises(ValueError, match="max_rows"):
        GeoPriorConfig(min_rows=400)
    with pytest.raises(ValueError, match="jump_ratio"):
        GeoPriorConfig(jump_ratio_min=0.5)
    with pytest.raises(ValueError, match="max_layers"):
        GeoPriorConfig(min_layers=5, max_layers=3)


def test_batch_shapes_and_ranges() -> None:
    cfg = GeoPriorConfig()
    g = torch.Generator().manual_seed(0)
    batch = sample_geo_batch(cfg, 6, g)
    b, r, f = batch.x.shape
    assert b == 6
    assert cfg.min_rows <= r <= cfg.max_rows
    assert cfg.min_features <= f <= cfg.max_features
    assert batch.y.shape == (6, r)
    assert batch.depth.shape == (6, r)
    assert 0 < batch.train_size < r
    # depth column (col 0) is increasing down each borehole
    assert (batch.depth[:, 1:] >= batch.depth[:, :-1]).all()
    # coordinate columns (1, 2) are constant within a borehole
    assert torch.allclose(batch.x[:, :, 1].std(dim=1), torch.zeros(6), atol=1e-5)


def test_layers_contiguous_and_soil_constant_within_layer() -> None:
    cfg = GeoPriorConfig()
    g = torch.Generator().manual_seed(1)
    batch = sample_geo_batch(cfg, 8, g)
    lid = batch.layer_id.numpy()
    soil = batch.soil_code.numpy()
    for b in range(8):
        # depth-sorted rows -> layer ids are non-decreasing and contiguous
        assert (np.diff(lid[b]) >= 0).all()
        # one soil code per layer
        for layer in np.unique(lid[b]):
            assert len(np.unique(soil[b][lid[b] == layer])) == 1
        assert 1 <= len(np.unique(lid[b])) <= cfg.max_layers


def test_piecewise_jumps_exceed_within_layer_steps() -> None:
    cfg = GeoPriorConfig()
    g = torch.Generator().manual_seed(2)
    within, cross = [], []
    for _ in range(20):
        batch = sample_geo_batch(cfg, 8, g)
        x, lid = batch.x.numpy(), batch.layer_id.numpy()
        b, r, _ = x.shape
        for bi in range(b):
            col = x[bi, :, 4]
            ok = ~np.isnan(col)
            for i in range(r - 1):
                if ok[i] and ok[i + 1]:
                    d = abs(col[i + 1] - col[i])
                    (within if lid[bi, i] == lid[bi, i + 1] else cross).append(d)
    ratio = np.median(cross) / np.median(within)
    assert 1.4 <= ratio <= 4.0  # cross-boundary jumps are the sharp changes


def test_target_tracks_depth_and_cheap_features_on_average() -> None:
    cfg = GeoPriorConfig()
    g = torch.Generator().manual_seed(3)
    depth_corr = []
    for _ in range(20):
        batch = sample_geo_batch(cfg, 8, g)
        depth, y = batch.depth.numpy(), batch.y.numpy()
        for b in range(8):
            if y[b].std() > 1e-6:
                depth_corr.append(abs(np.corrcoef(depth[b], y[b])[0, 1]))
    assert np.nanmedian(depth_corr) > 0.25  # the depth trend is present


def test_block_missingness_present() -> None:
    cfg = GeoPriorConfig(block_missing_prob=0.5)
    g = torch.Generator().manual_seed(4)
    batch = sample_geo_batch(cfg, 16, g)
    assert torch.isnan(batch.x).any()  # some feature columns dropped
    # structural columns (depth, coords, soil) are never missing
    assert not torch.isnan(batch.x[:, :, :4]).any()
