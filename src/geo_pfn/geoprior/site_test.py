"""Tests for the multi-borehole site sampler."""

from __future__ import annotations

import numpy as np
import torch

from geo_pfn.geoprior.config import GeoPriorConfig
from geo_pfn.geoprior.site import sample_geo_site_batch


def test_shapes_and_masks() -> None:
    cfg = GeoPriorConfig()
    g = torch.Generator().manual_seed(0)
    batch = sample_geo_site_batch(cfg, 6, g)
    b, r, f = batch.x.shape
    assert b == 6 and cfg.min_rows <= r <= cfg.max_rows
    assert batch.context_mask.shape == (6, r)
    assert batch.hole_id.shape == (6, r)
    assert batch.context_mask.dtype == torch.bool
    # structural columns never missing; some feature cells can be
    assert not torch.isnan(batch.x[:, :, :4]).any()
    # every site has both observed and query rows
    for i in range(6):
        assert batch.context_mask[i].any() and not batch.context_mask[i].all()


def test_p_single_extremes() -> None:
    g = torch.Generator().manual_seed(1)
    all_single = sample_geo_site_batch(GeoPriorConfig(p_single=1.0), 8, g)
    assert (all_single.hole_id == 0).all()  # every table is one hole
    all_multi = sample_geo_site_batch(GeoPriorConfig(p_single=0.0), 8, g)
    assert (all_multi.hole_id.max(dim=1).values >= 1).all()  # every table multi-hole


def test_target_holes_sparse_neighbors_dense() -> None:
    cfg = GeoPriorConfig(p_single=0.0)
    g = torch.Generator().manual_seed(2)
    batch = sample_geo_site_batch(cfg, 16, g)
    saw_sparse = saw_dense = False
    for b in range(16):
        hid, cm = batch.hole_id[b].numpy(), batch.context_mask[b].numpy()
        for h in np.unique(hid):
            frac = cm[hid == h].mean()
            n = int((hid == h).sum())
            if frac == 0.0 or (0 < cm[hid == h].sum() <= cfg.max_anchors):
                saw_sparse = saw_sparse or cm[hid == h].sum() <= cfg.max_anchors
            if frac == 1.0 and n > cfg.max_anchors:
                saw_dense = True
    assert saw_sparse and saw_dense  # both dense-neighbor and sparse-target holes exist


def test_nearby_holes_more_similar_than_far() -> None:
    cfg = GeoPriorConfig(p_single=0.0)
    g = torch.Generator().manual_seed(3)
    near, far = [], []
    for _ in range(40):
        batch = sample_geo_site_batch(cfg, 8, g)
        x, y, hid = batch.x.numpy(), batch.y.numpy(), batch.hole_id.numpy()
        for b in range(8):
            holes = np.unique(hid[b])
            if len(holes) < 3:
                continue
            hm = np.array([y[b][hid[b] == h].mean() for h in holes])
            hxy = np.array([x[b][hid[b] == h][0, 1:3] for h in holes])
            dists = [
                (np.hypot(*(hxy[i] - hxy[j])), abs(hm[i] - hm[j]))
                for i in range(len(holes))
                for j in range(i + 1, len(holes))
            ]
            med = np.median([d for d, _ in dists])
            for d, dy in dists:
                (near if d < med else far).append(dy)
    # nearby holes have smaller target differences (the spatial field is real)
    assert np.mean(near) < np.mean(far)
