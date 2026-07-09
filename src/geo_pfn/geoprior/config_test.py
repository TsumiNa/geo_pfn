"""Tests for GeoPriorConfig validation."""

from __future__ import annotations

import pytest

from geo_pfn.geoprior.config import GeoPriorConfig


def test_defaults() -> None:
    cfg = GeoPriorConfig()
    assert cfg.max_rows == 300
    assert cfg.max_features == 28


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_features": 1}, "min_features"),
        ({"spacing_mean": 0.0}, "spacing_mean"),
        ({"depth_trend_scale": -1.0}, "depth_trend_scale"),
        ({"n_soil_types": 1}, "n_soil_types"),
        ({"min_train_frac": 0.9, "max_train_frac": 0.5}, "train fractions"),
    ],
)
def test_rejects_invalid(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GeoPriorConfig(**kwargs)
