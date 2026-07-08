"""Tests for geo_pfn.haneda.runners."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from geo_pfn.haneda.runners import (
    ContextConfig,
    _stratified_context,
    classification_metrics,
    make_baseline,
    minipfn_predict_proba,
    regression_metrics,
)
from geo_pfn.minipfn.config import ModelConfig
from geo_pfn.minipfn.model import MiniPFN


@pytest.fixture()
def tiny_model() -> MiniPFN:
    torch.manual_seed(0)
    return MiniPFN(
        ModelConfig(d_model=32, n_layers=1, n_heads=2, max_classes=4, feature_emb_dim=8)
    )


def make_task(n_train: int = 60, n_test: int = 20, n_feat: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(n_train, n_feat))
    x_test = rng.normal(size=(n_test, n_feat))
    x_train[rng.random(x_train.shape) < 0.1] = np.nan
    y_train = rng.integers(0, 4, size=n_train)
    return x_train, y_train, x_test


def test_stratified_context_covers_all_classes() -> None:
    y = torch.tensor([0] * 50 + [1] * 5 + [2] * 3 + [3] * 2)
    generator = torch.Generator().manual_seed(0)
    idx = _stratified_context(y, 16, generator)
    assert len(idx) == 16
    assert len(set(idx.tolist())) == 16  # no duplicates
    assert set(y[idx].tolist()) == {0, 1, 2, 3}


def test_stratified_context_small_train_returns_all() -> None:
    y = torch.tensor([0, 1, 2])
    generator = torch.Generator().manual_seed(0)
    idx = _stratified_context(y, 16, generator)
    assert sorted(idx.tolist()) == [0, 1, 2]


def test_minipfn_predict_proba_shape_and_normalization(tiny_model) -> None:
    x_train, y_train, x_test = make_task()
    config = ContextConfig(ctx_size=32, n_ensembles=2, test_chunk=7)
    proba = minipfn_predict_proba(
        tiny_model, x_train, y_train, x_test, 4, config, 0, torch.device("cpu")
    )
    assert proba.shape == (20, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_minipfn_predict_proba_deterministic_and_chunk_invariant(tiny_model) -> None:
    x_train, y_train, x_test = make_task()
    kwargs = dict(n_classes=4, seed=7, device=torch.device("cpu"))
    small = ContextConfig(ctx_size=32, n_ensembles=2, test_chunk=3)
    big = ContextConfig(ctx_size=32, n_ensembles=2, test_chunk=100)
    p1 = minipfn_predict_proba(
        tiny_model, x_train, y_train, x_test, config=small, **kwargs
    )
    p2 = minipfn_predict_proba(
        tiny_model, x_train, y_train, x_test, config=small, **kwargs
    )
    p3 = minipfn_predict_proba(
        tiny_model, x_train, y_train, x_test, config=big, **kwargs
    )
    np.testing.assert_allclose(p1, p2)  # same seed -> identical
    np.testing.assert_allclose(p1, p3, atol=1e-5)  # test rows are independent


def test_context_config_validation() -> None:
    with pytest.raises(ValueError, match="ctx_size"):
        ContextConfig(ctx_size=1)
    with pytest.raises(ValueError, match="n_ensembles"):
        ContextConfig(n_ensembles=0)


def test_make_baseline_families() -> None:
    x = np.random.default_rng(0).normal(size=(40, 3))
    x[0, 0] = np.nan
    y_reg = x[:, 1] * 2 + 1
    y_clf = (x[:, 1] > 0).astype(int)
    for name in ("hgbt", "linear", "depth", "dummy"):
        make_baseline(name, "regression").fit(x, y_reg)
        make_baseline(name, "classification").fit(x, y_clf)
    with pytest.raises(ValueError, match="unknown baseline"):
        make_baseline("nope", "regression")


def test_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = regression_metrics(y, y)
    assert perfect["rmse"] == 0.0 and perfect["r2"] == 1.0
    off = regression_metrics(y, y + 1.0)
    assert off["mae"] == 1.0 and off["rmse"] == 1.0 and off["r2"] < 1.0

    clf = classification_metrics(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]))
    assert clf["accuracy"] == 0.75
    assert 0.0 < clf["macro_f1"] < 1.0
