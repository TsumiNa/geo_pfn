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
