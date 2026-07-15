"""Tests for sparse-grid context sampling (no model, no data file)."""

import numpy as np

from geo_pfn.haneda.sparse_grid import sample_context_boreholes


def _fake_bores(n_holes: int = 40, rows_per_hole: int = 15) -> np.ndarray:
    return np.repeat(np.arange(n_holes), rows_per_hole)


def test_sampled_fraction_reaches_target_with_whole_boreholes() -> None:
    bores = _fake_bores()
    rng = np.random.default_rng(0)
    picked = sample_context_boreholes(bores, 0.07, rng)
    rows = np.isin(bores, picked).sum()
    assert rows >= 0.07 * len(bores)
    # whole boreholes only: row count is a multiple of the per-hole size
    assert rows % 15 == 0
    # minimal: dropping the last picked hole would fall below the target
    assert rows - 15 < 0.07 * len(bores)


def test_sampling_is_deterministic_per_rng_state() -> None:
    bores = _fake_bores()
    a = sample_context_boreholes(bores, 0.08, np.random.default_rng(7))
    b = sample_context_boreholes(bores, 0.08, np.random.default_rng(7))
    assert a.tolist() == b.tolist()


def test_different_seeds_give_different_surveys() -> None:
    bores = _fake_bores()
    a = sample_context_boreholes(bores, 0.08, np.random.default_rng(1))
    b = sample_context_boreholes(bores, 0.08, np.random.default_rng(2))
    assert a.tolist() != b.tolist()
