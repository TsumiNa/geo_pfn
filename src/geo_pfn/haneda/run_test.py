"""Tests for geo_pfn.haneda.run configuration handling."""

from __future__ import annotations

import pytest

from geo_pfn.haneda.run import RunConfig, parse_context_scan


def test_run_config_defaults_valid() -> None:
    config = RunConfig()
    assert config.n_folds == 5
    assert set(config.experiments) == {"ablation", "imputation", "context"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_folds": 1}, "n_folds"),
        ({"n_bins": 5}, "n_bins"),
        ({"experiments": ("ablation", "nope")}, "unknown experiments"),
        ({"context_scan": ""}, "context_scan"),
    ],
)
def test_run_config_rejects_invalid(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RunConfig(**kwargs)


def test_parse_context_scan() -> None:
    configs = parse_context_scan("128:16,512:4,2048")
    assert [(c.ctx_size, c.n_ensembles) for c in configs] == [
        (128, 16),
        (512, 4),
        (2048, 1),
    ]
