"""Tests for geo_pfn.haneda.finetune."""

from __future__ import annotations

import inspect

import pytest

from geo_pfn.haneda.data import FeatureSet, Imputation
from geo_pfn.haneda.finetune import FinetuneConfig, make_finetuned_v2


def test_config_defaults_valid() -> None:
    config = FinetuneConfig()
    assert config.feature_set is FeatureSet.LCSG
    assert Imputation.NATIVE in config.imputations


def test_config_coerces_strings() -> None:
    config = FinetuneConfig(feature_set="LC", imputations=("native",))
    assert config.feature_set is FeatureSet.LC
    assert config.imputations == (Imputation.NATIVE,)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epochs": 0}, "epochs"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"n_folds": 1}, "n_folds"),
        ({"folds_subset": (5,)}, "folds_subset"),
        ({"grouped_val_frac": 0.6}, "grouped_val_frac"),
    ],
)
def test_config_rejects_invalid(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FinetuneConfig(**kwargs)


def test_make_finetuned_v2_forces_v2_checkpoint() -> None:
    model = make_finetuned_v2(device="cpu", epochs=1)
    source = inspect.getsource(type(model)._create_estimator)
    assert "ModelVersion.V2," in source  # not the license-restricted V2_5
    assert model.epochs == 1
