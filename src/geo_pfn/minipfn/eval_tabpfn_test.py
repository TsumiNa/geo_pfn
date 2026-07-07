"""Tests for the TabPFN v2 baseline harness (stubbed classifier, no weights)."""

import math

import numpy as np
import torch

from geo_pfn.minipfn.eval_tabpfn import _fill_train_means, tabpfn_task_accuracies


class _StubClassifier:
    """Predicts the train majority class; raises on demand for fallback tests."""

    def __init__(self, raise_on_fit: bool = False) -> None:
        self.raise_on_fit = raise_on_fit
        self.fit_calls = 0

    def fit(self, x, y):
        if self.raise_on_fit:
            raise ValueError("all features are constant")
        self.fit_calls += 1
        values, counts = np.unique(y, return_counts=True)
        self._majority = values[np.argmax(counts)]
        return self

    def predict(self, x):
        return np.full(len(x), self._majority)


def _toy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 12, 4, generator=g)
    # both train slices (first 8 rows) contain two classes
    y = torch.tensor([[0] * 7 + [1] * 5, [1, 1, 1, 0, 1, 0, 1, 1] + [0] * 4])
    return x, y


def test_tabpfn_task_accuracies_majority_stub() -> None:
    x, y = _toy_batch()
    fallbacks: list[int] = []
    accs = tabpfn_task_accuracies(
        _StubClassifier(), x, y, train_size=8, fallbacks=fallbacks
    )
    # task 0: train majority is 0, test rows are [1,1,1,1] -> 0.0
    # task 1: train majority is 1, test rows are [0,0,0,0] -> 0.0
    assert accs == [0.0, 0.0]
    assert fallbacks == []


def test_tabpfn_task_accuracies_fits_every_task() -> None:
    x, y = _toy_batch()
    clf = _StubClassifier()
    tabpfn_task_accuracies(clf, x, y, train_size=8, fallbacks=[])
    assert clf.fit_calls == 2


def test_tabpfn_task_accuracies_single_class_guard() -> None:
    x, y = _toy_batch()
    y[0, :8] = 1  # task 0 train slice has one class only -> no fit attempted
    clf = _StubClassifier()
    accs = tabpfn_task_accuracies(clf, x, y, train_size=8, fallbacks=[])
    assert accs[0] == 1.0  # predicts class 1, test rows are all 1
    assert clf.fit_calls == 1  # only task 1 reached the classifier


def test_tabpfn_task_accuracies_fallback_on_error() -> None:
    x, y = _toy_batch()
    fallbacks: list[int] = []
    accs = tabpfn_task_accuracies(
        _StubClassifier(raise_on_fit=True), x, y, train_size=8, fallbacks=fallbacks
    )
    assert len(accs) == 2 and all(0.0 <= a <= 1.0 for a in accs)
    assert fallbacks == [0, 1]


def test_fill_train_means_uses_train_stats_only() -> None:
    x = torch.zeros(1, 6, 2)
    x[0, :4, 0] = torch.tensor([1.0, 2.0, 3.0, math.nan])
    x[0, 4:, 0] = math.nan  # test rows missing in column 0
    x[0, :, 1] = 5.0
    filled = _fill_train_means(x, train_size=4)
    assert not torch.isnan(filled).any()
    assert torch.allclose(filled[0, 3:, 0], torch.full((3,), 2.0))  # mean of 1,2,3
    assert torch.equal(filled[0, :, 1], torch.full((6,), 5.0))
