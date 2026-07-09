"""Tests for geo_pfn.geopfn.predict."""

from __future__ import annotations

import numpy as np
import torch

from geo_pfn.geopfn.model import GeoPFN, GeoPFNConfig
from geo_pfn.geopfn.predict import EnsembleConfig, predict_geopfn


def test_predict_shape_and_scale() -> None:
    torch.manual_seed(0)
    model = GeoPFN(GeoPFNConfig(col_layers=1, row_layers=2, d_model=64, n_heads=4))
    rng = np.random.default_rng(0)
    x_ctx = rng.normal(size=(120, 8))
    y_ctx = rng.normal(50, 10, size=120)  # real-scale target
    x_q = rng.normal(size=(30, 8))
    pred = predict_geopfn(
        model,
        x_ctx,
        y_ctx,
        x_q,
        EnsembleConfig(ctx_size=64, n_ensembles=3),
        0,
        torch.device("cpu"),
    )
    assert pred.shape == (30,)
    # predictions land in the target's real range, not normalized space
    assert 0 < pred.mean() < 100


def test_keep_context_forced_into_every_draw() -> None:
    torch.manual_seed(0)
    model = GeoPFN(GeoPFNConfig(col_layers=1, row_layers=2, d_model=64, n_heads=4))
    rng = np.random.default_rng(1)
    x_ctx = rng.normal(size=(200, 6))
    y_ctx = rng.normal(size=200)
    x_q = rng.normal(size=(10, 6))
    # with a tiny ctx_size and required keep rows, it still runs and returns finite
    keep = np.array([0, 1, 2, 3, 4])
    pred = predict_geopfn(
        model,
        x_ctx,
        y_ctx,
        x_q,
        EnsembleConfig(ctx_size=16, n_ensembles=2),
        7,
        torch.device("cpu"),
        keep_context=keep,
    )
    assert pred.shape == (10,) and np.isfinite(pred).all()


def test_coherent_predict_shape_and_scale() -> None:
    import torch
    from geo_pfn.geopfn.predict import CoherentConfig, predict_geopfn_coherent

    torch.manual_seed(0)
    model = GeoPFN(GeoPFNConfig(col_layers=1, row_layers=2, d_model=64, n_heads=4))
    rng = np.random.default_rng(0)
    # 10 boreholes x 15 rows; cols: [depth, X, Y, feat]
    x, y, bores = [], [], []
    for b in range(10):
        xy = rng.normal(size=2) * 100
        for _ in range(15):
            x.append([rng.normal(), xy[0], xy[1], rng.normal()])
            y.append(rng.normal(50, 10))
            bores.append(b)
    x, y, bores = np.array(x), np.array(y), np.array(bores)
    x_query = np.column_stack(
        [
            rng.normal(size=8),
            np.full(8, x[0, 1]),
            np.full(8, x[0, 2]),
            rng.normal(size=8),
        ]
    )
    pred = predict_geopfn_coherent(
        model,
        x,
        y,
        bores,
        x_query,
        CoherentConfig(n_holes=4, n_candidates=6, n_ensembles=3),
        0,
        torch.device("cpu"),
    )
    assert pred.shape == (8,) and np.isfinite(pred).all()
    assert 0 < pred.mean() < 100  # real target units
