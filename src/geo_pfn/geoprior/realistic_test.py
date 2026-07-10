"""Tests for the geo-realistic cheap-feature block."""

from __future__ import annotations

import numpy as np
import torch

from geo_pfn.geoprior.config import GeoPriorConfig
from geo_pfn.geoprior.prior import sample_geo_batch
from geo_pfn.geoprior.realistic import load_stats, realistic_block


def _r2(x_cols: np.ndarray, y: np.ndarray) -> float:
    x = np.column_stack([np.ones(len(y)), x_cols])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return 1.0 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()


def test_block_shapes_finite_and_deterministic() -> None:
    cfg = GeoPriorConfig()
    soft = torch.randn(4, 60, generator=torch.Generator().manual_seed(7))
    signal = torch.randn(4, 60, generator=torch.Generator().manual_seed(8))
    depth = torch.randn(4, 60, generator=torch.Generator().manual_seed(9))
    a = realistic_block(cfg, soft, signal, depth, 10, torch.Generator().manual_seed(0))
    b = realistic_block(cfg, soft, signal, depth, 10, torch.Generator().manual_seed(0))
    feats, target = a
    assert feats.shape == (4, 60, 10)
    assert target.shape == (4, 60)
    assert torch.isfinite(feats).all() and torch.isfinite(target).all()
    assert torch.allclose(feats, b[0]) and torch.allclose(target, b[1])


def test_cheap_cluster_is_collinear_and_marginals_realistic() -> None:
    cfg = GeoPriorConfig(p_geo_realistic=1.0)
    g = torch.Generator().manual_seed(0)
    stats = load_stats()
    pc1_fracs, wn_ok = [], []
    for _ in range(20):
        batch = sample_geo_batch(cfg, 8, g)
        feats = batch.x[:, :, 4 : 4 + 6].numpy()  # cheap cluster
        for bi in range(feats.shape[0]):
            col = feats[bi]
            if np.isnan(col).any():
                continue
            fs = (col - col.mean(0)) / (col.std(0) + 1e-9)
            sv = np.linalg.svd(fs - fs.mean(0), compute_uv=False)
            pc1_fracs.append(sv[0] ** 2 / (sv**2).sum())
            # Wn (col 0) stays inside the real observed range
            wn = col[:, 0]
            q = stats["cheap_quantiles"]["Wn"]
            wn_ok.append((wn.min() >= q[0] - 1e-3) and (wn.max() <= q[-1] + 1e-3))
    # dominant collinear axis (real cluster: 0.765); loose floor for the family
    assert np.mean(pc1_fracs) > 0.55
    assert np.mean(wn_ok) > 0.95


def test_target_depth_dominated_with_weak_feature_signal() -> None:
    cfg = GeoPriorConfig(p_geo_realistic=1.0)
    g = torch.Generator().manual_seed(1)
    dep, inc, soft_partial = [], [], []
    for _ in range(30):
        batch = sample_geo_batch(cfg, 8, g)
        x, y, depth = batch.x.numpy(), batch.y.numpy(), batch.depth.numpy()
        for bi in range(x.shape[0]):
            feats = x[bi, :, 4 : 4 + 6]
            if np.isnan(feats).any() or y[bi].std() < 1e-6:
                continue
            z, t = depth[bi], y[bi]
            dep.append(_r2(z[:, None], t))
            inc.append(_r2(np.column_stack([z, feats]), t))
            fs = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
            pc1_score = (fs - fs.mean(0)) @ np.linalg.svd(
                fs - fs.mean(0), full_matrices=False
            )[2][0]
            rp = pc1_score - np.polyval(np.polyfit(z, pc1_score, 1), z)
            rt = t - np.polyval(np.polyfit(z, t, 1), z)
            soft_partial.append(np.corrcoef(rp, rt)[0, 1])  # signed
    dep, inc = np.array(dep), np.array(inc)
    # depth dominates; cheap features add a small positive increment (real +0.12)
    assert dep.mean() > 0.4
    assert 0.02 < (inc - dep).mean() < 0.25
    # the dominant collinear axis carries no *systematic* Su signal beyond depth:
    # per-table partial corr is random-signed finite-sample noise, averaging to ~0
    # (real full-sample value +0.019). Per-table magnitude is not the property.
    assert abs(np.mean(soft_partial)) < 0.06


def test_p_geo_realistic_zero_recovers_generic() -> None:
    # with the realistic branch off, features are the random-MLP block (unbounded)
    cfg = GeoPriorConfig(p_geo_realistic=0.0)
    g = torch.Generator().manual_seed(2)
    batch = sample_geo_batch(cfg, 8, g)
    assert batch.x.shape[0] == 8 and torch.isfinite(batch.y).all()
