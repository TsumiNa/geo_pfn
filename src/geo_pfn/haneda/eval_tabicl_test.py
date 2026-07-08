"""Tests for geo_pfn.haneda.eval_tabicl."""

from __future__ import annotations

import pytest

from geo_pfn.haneda.data import FeatureSet, Imputation
from geo_pfn.haneda.eval_tabicl import run


def test_module_imports_without_tabicl() -> None:
    # tabicl is an ephemeral overlay dependency; the module itself must import
    # (and fail lazily inside run()) when tabicl is not installed.
    import geo_pfn.haneda.eval_tabicl as mod

    assert callable(mod.main)


def test_run_requires_tabicl_before_touching_data() -> None:
    with pytest.raises(ModuleNotFoundError, match="tabicl"):
        run(
            data_path="does-not-exist.csv",
            out_path="unused.json",
            device="cpu",
            feature_set=FeatureSet.LCSG,
            imputations=(Imputation.NATIVE,),
            n_folds=5,
            seed=42,
            n_bins=4,
        )
