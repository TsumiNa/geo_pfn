"""Tests for geo_pfn.haneda.infill."""

from __future__ import annotations

import numpy as np
import pytest

from geo_pfn.haneda.data import FeatureSet
from geo_pfn.haneda.data_test import make_df
from geo_pfn.haneda.infill import InfillConfig, depth_split


def test_config_validation() -> None:
    assert InfillConfig().feature_set is FeatureSet.LCSG
    with pytest.raises(ValueError, match="query_frac"):
        InfillConfig(query_frac=0.0)
    with pytest.raises(ValueError, match="n_folds"):
        InfillConfig(n_folds=1)


def test_depth_split_deep_is_query_shallow_is_context() -> None:
    df = make_df(n=80, seed=1)
    test_idx = np.arange(len(df))
    shallow, query = depth_split(df, test_idx, query_frac=0.5)
    # disjoint and drawn only from eligible boreholes
    assert not set(shallow) & set(query)
    depth = df["depth_m"].to_numpy()
    bores = df["BorSeq"].to_numpy()
    for bore in np.unique(bores):
        rows = test_idx[bores == bore]
        if len(rows) < 4:
            # too-small boreholes contribute nothing
            assert not (set(rows) & (set(shallow) | set(query)))
            continue
        q = [r for r in query if bores[r] == bore]
        s = [r for r in shallow if bores[r] == bore]
        assert q and s
        # every query row is deeper (more negative) than every shallow row
        assert max(depth[q]) <= min(depth[s])


def test_depth_split_respects_query_frac() -> None:
    df = make_df(n=120, seed=2)
    test_idx = np.arange(len(df))
    bores = df["BorSeq"].to_numpy()
    _, query = depth_split(df, test_idx, query_frac=0.25)
    _, query_big = depth_split(df, test_idx, query_frac=0.75)
    # a larger query fraction never yields fewer query rows
    assert len(query_big) >= len(query)
    # at least one shallow row is always kept per eligible borehole
    for bore in np.unique(bores):
        rows = test_idx[bores == bore]
        if len(rows) >= 4:
            assert sum(1 for r in query_big if bores[r] == bore) <= len(rows) - 1
