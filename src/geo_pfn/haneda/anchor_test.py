"""Tests for geo_pfn.haneda.anchor."""

from __future__ import annotations

import numpy as np
import pytest

from geo_pfn.haneda.data_test import make_df
from geo_pfn.haneda.anchor import AnchorConfig, select_anchors


def test_config_validation() -> None:
    assert AnchorConfig().k_anchors == (1, 2, 3, 5)
    with pytest.raises(ValueError, match="k_anchors"):
        AnchorConfig(k_anchors=(0,))
    with pytest.raises(ValueError, match="n_folds"):
        AnchorConfig(n_folds=1)


def test_select_anchors_disjoint_and_depth_spread() -> None:
    df = make_df(n=120, seed=3)
    test_idx = np.arange(len(df))
    gen = np.random.default_rng(0)
    k = 3
    anchor, query = select_anchors(df, test_idx, k, gen)
    assert not set(anchor) & set(query)
    depth = df["depth_m"].to_numpy()
    bores = df["BorSeq"].to_numpy()
    for bore in np.unique(bores):
        rows = test_idx[bores == bore]
        a = [r for r in anchor if bores[r] == bore]
        if len(rows) < 2 * k:
            assert not a  # too small: dropped
            continue
        assert len(a) == k
        # anchors span the depth range: their spread covers most of the borehole
        span = depth[rows].max() - depth[rows].min()
        a_span = depth[a].max() - depth[a].min()
        assert a_span >= 0.4 * span


def test_select_anchors_more_k_more_anchors() -> None:
    df = make_df(n=160, seed=4)
    test_idx = np.arange(len(df))
    gen = np.random.default_rng(1)
    n1 = len(select_anchors(df, test_idx, 1, gen)[0])
    n3 = len(select_anchors(df, test_idx, 3, gen)[0])
    assert n3 >= n1


def test_select_anchors_reproducible() -> None:
    df = make_df(n=100, seed=5)
    test_idx = np.arange(len(df))
    a1, q1 = select_anchors(df, test_idx, 2, np.random.default_rng(7))
    a2, q2 = select_anchors(df, test_idx, 2, np.random.default_rng(7))
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(q1, q2)
