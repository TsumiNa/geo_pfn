"""Tests for the synthetic prior and the feature-dropout corruptions."""

import math

import pytest
import torch

from geo_pfn.minipfn.config import PriorConfig
from geo_pfn.minipfn.prior import (
    apply_cell_missing,
    apply_test_missing,
    random_cell_missing,
    random_test_missing,
    sample_dropped_columns,
    sample_prior_batch,
)


@pytest.fixture
def cfg() -> PriorConfig:
    return PriorConfig(min_rows=40, max_rows=80)


def test_batch_shapes_and_ranges(cfg: PriorConfig) -> None:
    g = torch.Generator().manual_seed(0)
    batch = sample_prior_batch(cfg, batch_size=8, generator=g)
    b, r, f = batch.x.shape
    assert b == 8
    assert cfg.min_rows <= r <= cfg.max_rows
    assert cfg.min_features <= f <= cfg.max_features
    assert 0 < batch.train_size < r
    assert batch.x.dtype == torch.float32
    assert batch.y.dtype == torch.int64
    assert bool((batch.n_classes >= cfg.min_classes).all())
    assert bool((batch.n_classes <= cfg.max_classes).all())
    assert bool((batch.y >= 0).all())
    assert bool((batch.y < batch.n_classes.unsqueeze(1)).all())


def test_all_classes_present_and_x_mostly_finite(cfg: PriorConfig) -> None:
    g = torch.Generator().manual_seed(1)
    batch = sample_prior_batch(cfg, batch_size=16, generator=g)
    for b in range(16):
        present = batch.y[b].unique().numel()
        assert present == int(batch.n_classes[b])
    finite_frac = torch.isfinite(batch.x).float().mean()
    assert finite_frac > 1.0 - cfg.max_cell_missing_frac - 0.05
    assert not torch.isinf(batch.x).any()


def test_determinism(cfg: PriorConfig) -> None:
    b1 = sample_prior_batch(cfg, 4, torch.Generator().manual_seed(7))
    b2 = sample_prior_batch(cfg, 4, torch.Generator().manual_seed(7))
    assert torch.equal(b1.x.nan_to_num(), b2.x.nan_to_num())
    assert torch.equal(b1.y, b2.y)
    assert b1.train_size == b2.train_size


def test_sample_dropped_columns_counts() -> None:
    g = torch.Generator().manual_seed(0)
    n_drop = torch.tensor([0, 1, 3, 5])
    dropped = sample_dropped_columns(4, 6, n_drop, g)
    assert dropped.shape == (4, 6)
    assert torch.equal(dropped.sum(dim=1), n_drop)


def test_apply_test_missing_only_touches_test_rows() -> None:
    x = torch.zeros(2, 10, 4)
    dropped = torch.tensor([[True, False, True, False], [False, False, False, True]])
    out = apply_test_missing(x, train_size=6, dropped=dropped)
    assert not torch.isnan(out[:, :6]).any()
    assert torch.isnan(out[0, 6:, 0]).all() and torch.isnan(out[0, 6:, 2]).all()
    assert not torch.isnan(out[0, 6:, 1]).any()
    assert torch.isnan(out[1, 6:, 3]).all()
    assert not torch.isnan(x).any()  # input untouched


def test_random_test_missing_noop_when_prob_zero() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 12, 5, generator=g)
    out = random_test_missing(
        x, train_size=8, task_prob=0.0, max_drop_frac=0.5, generator=g
    )
    assert torch.equal(out, x)


def test_apply_cell_missing_keeps_one_observed_per_row() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 10, 5, generator=g)
    out = apply_cell_missing(x, torch.full((4, 10, 5), 0.99), g)
    assert (~torch.isnan(out)).any(dim=-1).all()  # no fully blank row
    assert torch.isnan(out).float().mean() > 0.5


def test_random_cell_missing_noop_when_prob_zero() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 12, 5, generator=g)
    y = torch.randint(0, 2, (3, 12), generator=g)
    out = random_cell_missing(x, y, torch.full((3,), 2), 0.0, 0.5, 0.3, g)
    assert torch.equal(out, x)


def test_random_cell_missing_corrupts_train_and_test_rows() -> None:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(16, 40, 6, generator=g)
    y = torch.randint(0, 2, (16, 40), generator=g)
    out = random_cell_missing(x, y, torch.full((16,), 2), 1.0, 0.6, 0.0, g)
    assert torch.isnan(out[:, :20]).any()  # context rows corrupted too
    assert torch.isnan(out[:, 20:]).any()
    assert (~torch.isnan(out)).any(dim=-1).all()


def test_random_cell_missing_informative_correlates_with_class() -> None:
    g = torch.Generator().manual_seed(2)
    x = torch.randn(16, 200, 8, generator=g)
    y = torch.randint(0, 2, (16, 200), generator=g)
    out = random_cell_missing(x, y, torch.full((16,), 2), 1.0, 0.5, 1.0, g)
    missing_frac = torch.isnan(out).float().mean(dim=-1)  # (B, R)
    frac_class1 = missing_frac[y == 1].mean()
    frac_class0 = missing_frac[y == 0].mean()
    assert frac_class1 - frac_class0 > 0.05


def test_random_test_missing_bounds() -> None:
    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, 12, 6, generator=g)
    out = random_test_missing(
        x, train_size=8, task_prob=1.0, max_drop_frac=0.5, generator=g
    )
    assert not torch.isnan(out[:, :8]).any()
    dropped_per_task = torch.isnan(out[:, 8:]).all(dim=1).sum(dim=1)
    assert bool((dropped_per_task >= 1).all())
    assert bool((dropped_per_task <= math.floor(6 * 0.5)).all())
