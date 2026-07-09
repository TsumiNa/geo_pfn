"""Tests for geo_pfn.haneda.runners."""

from __future__ import annotations

import numpy as np
import pytest

from geo_pfn.haneda.runners import (
    classification_metrics,
    make_baseline,
    regression_metrics,
)


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
