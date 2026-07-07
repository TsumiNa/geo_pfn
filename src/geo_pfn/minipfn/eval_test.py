"""Tests for the evaluation harness (strategies, pairing, accuracy ranges)."""

import torch

from geo_pfn.minipfn.config import ModelConfig, PriorConfig
from geo_pfn.minipfn.eval import (
    evaluate_real,
    evaluate_synthetic,
    evaluate_synthetic_cells,
)
from geo_pfn.minipfn.model import MiniPFN


def _tiny_model() -> MiniPFN:
    torch.manual_seed(0)
    return MiniPFN(
        ModelConfig(d_model=32, n_layers=2, n_heads=2, feature_emb_dim=8)
    ).eval()


def test_evaluate_synthetic_structure() -> None:
    model = _tiny_model()
    prior_cfg = PriorConfig(min_rows=30, max_rows=40, min_features=4, max_features=6)
    drop_fracs = [0.5]
    results = evaluate_synthetic(
        model,
        prior_cfg,
        torch.device("cpu"),
        n_batches=2,
        batch_size=3,
        seed=0,
        drop_fracs=drop_fracs,
    )
    expected = {
        "minipfn nan-fill",
        "minipfn drop-context",
        "minipfn mean-impute",
        "logreg impute",
        "logreg retrain",
    }
    assert expected <= set(results)
    for strategy in expected:
        for frac in [0.0, 0.5]:
            accs = results[strategy][frac]
            assert len(accs) == 6  # one accuracy per task, paired across strategies
            assert all(0.0 <= a <= 1.0 for a in accs)


def test_evaluate_synthetic_cells_structure() -> None:
    model = _tiny_model()
    prior_cfg = PriorConfig(min_rows=30, max_rows=40, min_features=4, max_features=6)
    mcar, mnar = evaluate_synthetic_cells(
        model,
        prior_cfg,
        torch.device("cpu"),
        n_batches=2,
        batch_size=3,
        seed=0,
        miss_rates=[0.4],
        mnar_rate=0.3,
    )
    expected = {
        "minipfn native",
        "minipfn mean-impute",
        "logreg mean-impute",
        "logreg mean+indicator",
        "logreg knn-impute",
    }
    assert expected <= set(mcar) and expected <= set(mnar)
    for strategy in expected:
        for rate in [0.0, 0.4]:
            accs = mcar[strategy][rate]
            assert len(accs) == 6
            assert all(0.0 <= a <= 1.0 for a in accs)
        assert len(mnar[strategy][0.3]) == 6


def test_evaluate_real_runs() -> None:
    model = _tiny_model()
    results = evaluate_real(
        model,
        torch.device("cpu"),
        seed=0,
        drop_fracs=[0.25],
        max_features=6,
        n_repeats=2,
    )
    assert set(results) == {"breast_cancer", "wine"}
    for dataset_results in results.values():
        accs = dataset_results["minipfn nan-fill"][0.25]
        assert len(accs) == 2
        assert all(0.0 <= a <= 1.0 for a in accs)
