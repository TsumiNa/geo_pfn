"""Paired TabPFN v2 baseline: ``python -m geo_pfn.minipfn.eval_tabpfn``.

Replays the exact task and corruption sequence of ``eval --scenario cells``
(same seed → same synthetic tasks, same missing cells) with the original
TabPFN v2 checkpoint as the predictor, so its numbers are directly comparable
with the mini-PFN tables. Two strategies per condition:

- ``tabpfn-v2 native``      — NaN cells go straight into TabPFN (its own
  missing-indicator + train-mean-imputation encoder handles them).
- ``tabpfn-v2 mean-impute`` — every NaN replaced by the train column mean
  before fit/predict, hiding the missingness pattern from the model.

``logreg mean-impute`` is recomputed on the corrupted tensors as a pairing
check: its numbers must reproduce the mini-PFN evaluation tables exactly.

Uses ``ModelVersion.V2`` weights — the release whose license permits
commercial-friendly use and that downloads without a login.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from geo_pfn.minipfn.config import PriorConfig
from geo_pfn.minipfn.eval import (
    Results,
    _logreg_accuracy,
    _train_col_means,
    corrupt_informative,
    corrupt_mcar_rows,
    print_table,
)
from geo_pfn.minipfn.prior import sample_prior_batch
from geo_pfn.minipfn.train import resolve_device


def make_tabpfn_v2(device: str, n_estimators: int | None):
    """Build a TabPFN v2 classifier (imported lazily; downloads weights on first use)."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    overrides: dict = {"device": device}
    if n_estimators is not None:
        overrides["n_estimators"] = n_estimators
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, **overrides)


def tabpfn_task_accuracies(
    clf, x: torch.Tensor, y: torch.Tensor, train_size: int, fallbacks: list[int]
) -> list[float]:
    """Per-task test accuracy of TabPFN on one batch (one fit/predict per task)."""
    accs = []
    for b in range(x.shape[0]):
        x_train = x[b, :train_size].numpy()
        y_train = y[b, :train_size].numpy()
        x_test = x[b, train_size:].numpy()
        y_test = y[b, train_size:].numpy()
        classes = np.unique(y_train)
        if len(classes) < 2:  # degenerate train slice: predict the only seen class
            accs.append(float((y_test == classes[0]).mean()))
            continue
        try:
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
        except ValueError:  # e.g. every feature constant/all-NaN after corruption
            fallbacks.append(b)
            majority = classes[np.argmax([(y_train == c).sum() for c in classes])]
            accs.append(float((y_test == majority).mean()))
            continue
        accs.append(float((pred == y_test).mean()))
    return accs


def _fill_train_means(x: torch.Tensor, train_size: int) -> torch.Tensor:
    means = _train_col_means(x, train_size)
    return torch.where(torch.isnan(x), means.unsqueeze(1).expand_as(x), x)


def _eval_condition(
    clf,
    x_corrupt: torch.Tensor,
    y: torch.Tensor,
    train_size: int,
    rate: float,
    results: Results,
    fallbacks: list[int],
    pairing_check: bool,
) -> None:
    results["tabpfn-v2 native"][rate] += tabpfn_task_accuracies(
        clf, x_corrupt, y, train_size, fallbacks
    )
    x_imputed = _fill_train_means(x_corrupt, train_size)
    results["tabpfn-v2 mean-impute"][rate] += tabpfn_task_accuracies(
        clf, x_imputed, y, train_size, fallbacks
    )
    if pairing_check:
        x_np = x_corrupt.numpy()
        y_np = y.numpy()
        for b in range(x_corrupt.shape[0]):
            results["logreg mean-impute"][rate].append(
                _logreg_accuracy(
                    x_np[b, :train_size],
                    y_np[b, :train_size],
                    x_np[b, train_size:],
                    y_np[b, train_size:],
                )
            )


def evaluate_synthetic_tabpfn(
    clf,
    prior_cfg: PriorConfig,
    n_batches: int,
    batch_size: int,
    seed: int,
    miss_rates: list[float],
    mnar_rate: float,
) -> tuple[Results, Results, list[int]]:
    """Mirror of ``evaluate_synthetic_cells``: identical generator consumption order."""
    generator = torch.Generator().manual_seed(seed)
    mcar: Results = defaultdict(lambda: defaultdict(list))
    mnar: Results = defaultdict(lambda: defaultdict(list))
    fallbacks: list[int] = []
    t_start = time.time()
    for i in range(n_batches):
        batch = sample_prior_batch(prior_cfg, batch_size, generator)
        for rate in [0.0, *miss_rates]:
            x_c = corrupt_mcar_rows(batch.x, rate, generator)
            _eval_condition(
                clf, x_c, batch.y, batch.train_size, rate, mcar, fallbacks, True
            )
        x_i = corrupt_informative(
            batch.x, batch.y, batch.n_classes, mnar_rate, generator
        )
        _eval_condition(
            clf, x_i, batch.y, batch.train_size, mnar_rate, mnar, fallbacks, True
        )
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (n_batches - i - 1)
        print(
            f"batch {i + 1}/{n_batches}  elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m",
            flush=True,
        )
    return mcar, mnar, fallbacks


def evaluate_real_tabpfn(
    clf,
    seed: int,
    miss_rates: list[float],
    max_features: int,
    n_repeats: int = 5,
) -> dict[str, Results]:
    """Mirror of ``evaluate_real_cells`` with TabPFN as the predictor."""
    from sklearn.datasets import load_breast_cancer, load_wine

    all_results: dict[str, Results] = {}
    fallbacks: list[int] = []
    for name, loader in (("breast_cancer", load_breast_cancer), ("wine", load_wine)):
        data_x, data_y = loader(return_X_y=True)
        rng = np.random.default_rng(seed)
        row_order = rng.permutation(len(data_x))[:300]
        col_order = rng.permutation(data_x.shape[1])[:max_features]
        x = torch.tensor(
            data_x[row_order][:, col_order], dtype=torch.float32
        ).unsqueeze(0)
        y = torch.tensor(data_y[row_order], dtype=torch.long).unsqueeze(0)

        results: Results = defaultdict(lambda: defaultdict(list))
        generator = torch.Generator().manual_seed(seed)
        for _ in range(n_repeats):
            for rate in [0.0, *miss_rates]:
                x_c = corrupt_mcar_rows(x, rate, generator)
                _eval_condition(clf, x_c, y, 100, rate, results, fallbacks, False)
        all_results[name] = results
        print(f"real data done: {name}", flush=True)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--miss-rates", type=str, default="0.2,0.4,0.6")
    parser.add_argument("--mnar-rate", type=float, default=0.3)
    parser.add_argument("--real", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    clf = make_tabpfn_v2(str(device), args.n_estimators)
    print(f"tabpfn v2 on {device} n_estimators={clf.n_estimators}", flush=True)
    prior_cfg = PriorConfig()
    miss_rates = [float(f) for f in args.miss_rates.split(",") if f]
    n_tasks = args.n_batches * args.batch_size

    mcar, mnar, fallbacks = evaluate_synthetic_tabpfn(
        clf,
        prior_cfg,
        args.n_batches,
        args.batch_size,
        args.seed,
        miss_rates,
        args.mnar_rate,
    )
    print_table(f"synthetic, per-row MCAR ({n_tasks} tasks)", mcar, miss_rates, "miss")
    print_table(
        f"synthetic, label-informative MNAR at {args.mnar_rate:.0%} ({n_tasks} tasks)",
        mnar,
        [args.mnar_rate],
        "miss",
    )
    if fallbacks:
        print(f"\n{len(fallbacks)} task evaluations fell back to majority class")

    if args.real:
        for name, results in evaluate_real_tabpfn(
            clf, args.seed, miss_rates, prior_cfg.max_features
        ).items():
            print_table(f"real data, per-row MCAR: {name}", results, miss_rates, "miss")


if __name__ == "__main__":
    main()
