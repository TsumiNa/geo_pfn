"""Evaluate missing-feature strategies: ``python -m geo_pfn.minipfn.eval``.

Two scenarios, selected with ``--scenario``:

``cells`` (default, questionnaire-style): every row — train context and test
alike — loses a random, per-row-varying fraction of its cells, like surveys
with different completion rates. Also reports a label-informative (MNAR)
table where *which* cells are missing correlates with the class, so the
missingness pattern itself carries signal. Strategies:

- ``minipfn native``       — NaN cells go straight into the model (missing flag).
- ``minipfn mean-impute``  — every NaN replaced by the train column mean and
  presented as observed (flag hidden from the model).
- ``logreg mean-impute``   — SimpleImputer(mean) + logistic regression.
- ``logreg mean+indicator``— same, plus missing-indicator columns (the classical
  way to keep the missingness signal).
- ``logreg knn-impute``    — KNNImputer(5) + logistic regression.

``columns``: whole feature columns are hidden from the TEST rows only (the
train context stays complete); strategies additionally include
``drop-context`` (remove the columns from both sides — free for in-context
learners) and ``logreg retrain``.

Rate/fraction 0.0 is each family's clean-data reference.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geo_pfn.minipfn.config import PriorConfig
from geo_pfn.minipfn.model import MiniPFN, mask_class_logits
from geo_pfn.minipfn.prior import (
    apply_cell_missing,
    apply_test_missing,
    sample_prior_batch,
)
from geo_pfn.minipfn.train import load_checkpoint, resolve_device

Results = dict[str, dict[float, list[float]]]


def model_task_accuracies(
    model: MiniPFN,
    x: torch.Tensor,
    y: torch.Tensor,
    train_size: int,
    n_classes: torch.Tensor,
    device: torch.device,
) -> list[float]:
    """Per-task test accuracy of the in-context model on one batch."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device), y.to(device), train_size)
        logits = mask_class_logits(logits, n_classes.to(device))
        correct = logits.argmax(-1).cpu() == y[:, train_size:]
    return correct.float().mean(dim=1).tolist()


def _train_col_means(x: torch.Tensor, train_size: int) -> torch.Tensor:
    """(B, F) NaN-aware column means over the train rows (0 for all-NaN columns)."""
    train_x = x[:, :train_size]
    observed = ~torch.isnan(train_x)
    zero = torch.zeros((), dtype=x.dtype)
    return torch.where(observed, train_x, zero).sum(dim=1) / observed.sum(dim=1).clamp(
        min=1
    )


def _logreg_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    imputer: SimpleImputer | KNNImputer | None = None,
) -> float:
    if imputer is None:
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    pipeline = make_pipeline(
        imputer, StandardScaler(), LogisticRegression(max_iter=500)
    )
    pipeline.fit(x_train, y_train)
    return float((pipeline.predict(x_test) == y_test).mean())


def corrupt_mcar_rows(
    x: torch.Tensor, mean_rate: float, generator: torch.Generator
) -> torch.Tensor:
    """Questionnaire-style MCAR: every row loses a rate ~ U[0, 2*mean_rate] of cells."""
    if mean_rate <= 0:
        return x
    b, r, _ = x.shape
    rates = torch.rand(b, r, generator=generator) * 2.0 * mean_rate
    return apply_cell_missing(
        x, rates.clamp(max=0.95).unsqueeze(-1).expand_as(x), generator
    )


def corrupt_informative(
    x: torch.Tensor,
    y: torch.Tensor,
    n_classes: torch.Tensor,
    base_rate: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Label-informative (MNAR) missingness on all rows.

    A random half of the columns gets missing probability ``2 * base_rate *
    class_rank`` (top class ~2x base, bottom class ~0), the rest ``base_rate``
    — so the missingness pattern itself predicts the label.
    """
    b, r, f = x.shape
    info_cols = torch.rand(b, 1, f, generator=generator) < 0.5
    y_rank = y.float() / (n_classes.view(-1, 1).float() - 1.0).clamp(min=1.0)
    scaled = (2.0 * base_rate * y_rank).unsqueeze(-1).expand(b, r, f)
    prob = torch.where(info_cols, scaled, torch.full_like(scaled, base_rate))
    return apply_cell_missing(x, prob.clamp(max=0.95), generator)


_CELL_IMPUTERS = {
    "logreg mean-impute": lambda: SimpleImputer(
        strategy="mean", keep_empty_features=True
    ),
    "logreg mean+indicator": lambda: SimpleImputer(
        strategy="mean", keep_empty_features=True, add_indicator=True
    ),
    "logreg knn-impute": lambda: KNNImputer(n_neighbors=5, keep_empty_features=True),
}


def evaluate_batch_cells(
    model: MiniPFN,
    x_corrupt: torch.Tensor,
    y: torch.Tensor,
    train_size: int,
    n_classes: torch.Tensor,
    device: torch.device,
    rate: float,
    results: Results,
) -> None:
    """Evaluate all cell-scenario strategies on one corrupted batch."""
    results["minipfn native"][rate] += model_task_accuracies(
        model, x_corrupt, y, train_size, n_classes, device
    )

    means = _train_col_means(x_corrupt, train_size)  # (B, F)
    fill = means.unsqueeze(1).expand_as(x_corrupt)
    x_imputed = torch.where(torch.isnan(x_corrupt), fill, x_corrupt)
    results["minipfn mean-impute"][rate] += model_task_accuracies(
        model, x_imputed, y, train_size, n_classes, device
    )

    x_np = x_corrupt.numpy()
    y_np = y.numpy()
    for name, make_imputer in _CELL_IMPUTERS.items():
        for b in range(x_corrupt.shape[0]):
            results[name][rate].append(
                _logreg_accuracy(
                    x_np[b, :train_size],
                    y_np[b, :train_size],
                    x_np[b, train_size:],
                    y_np[b, train_size:],
                    imputer=make_imputer(),
                )
            )


def evaluate_synthetic_cells(
    model: MiniPFN,
    prior_cfg: PriorConfig,
    device: torch.device,
    n_batches: int,
    batch_size: int,
    seed: int,
    miss_rates: list[float],
    mnar_rate: float,
) -> tuple[Results, Results]:
    """Returns (MCAR results by rate, MNAR results at ``mnar_rate``)."""
    generator = torch.Generator().manual_seed(seed)
    mcar: Results = defaultdict(lambda: defaultdict(list))
    mnar: Results = defaultdict(lambda: defaultdict(list))
    for _ in range(n_batches):
        batch = sample_prior_batch(prior_cfg, batch_size, generator)
        for rate in [0.0, *miss_rates]:
            x_c = corrupt_mcar_rows(batch.x, rate, generator)
            evaluate_batch_cells(
                model,
                x_c,
                batch.y,
                batch.train_size,
                batch.n_classes,
                device,
                rate,
                mcar,
            )
        x_i = corrupt_informative(
            batch.x, batch.y, batch.n_classes, mnar_rate, generator
        )
        evaluate_batch_cells(
            model,
            x_i,
            batch.y,
            batch.train_size,
            batch.n_classes,
            device,
            mnar_rate,
            mnar,
        )
    return mcar, mnar


def evaluate_real_cells(
    model: MiniPFN,
    device: torch.device,
    seed: int,
    miss_rates: list[float],
    max_features: int,
    n_repeats: int = 5,
) -> dict[str, Results]:
    """Real small datasets with synthetic questionnaire-style corruption."""
    all_results: dict[str, Results] = {}
    for name, loader in (("breast_cancer", load_breast_cancer), ("wine", load_wine)):
        data_x, data_y = loader(return_X_y=True)
        rng = np.random.default_rng(seed)
        row_order = rng.permutation(len(data_x))[:300]
        col_order = rng.permutation(data_x.shape[1])[:max_features]
        x = torch.tensor(
            data_x[row_order][:, col_order], dtype=torch.float32
        ).unsqueeze(0)
        y = torch.tensor(data_y[row_order], dtype=torch.long).unsqueeze(0)
        n_classes = torch.tensor([int(data_y.max()) + 1])

        results: Results = defaultdict(lambda: defaultdict(list))
        generator = torch.Generator().manual_seed(seed)
        for _ in range(n_repeats):
            for rate in [0.0, *miss_rates]:
                x_c = corrupt_mcar_rows(x, rate, generator)
                evaluate_batch_cells(
                    model, x_c, y, 100, n_classes, device, rate, results
                )
        all_results[name] = results
    return all_results


def evaluate_batch(
    model: MiniPFN,
    x: torch.Tensor,
    y: torch.Tensor,
    train_size: int,
    n_classes: torch.Tensor,
    device: torch.device,
    drop_fracs: list[float],
    generator: torch.Generator,
    results: Results,
) -> None:
    """Evaluate all strategies on one batch of tasks, accumulating into ``results``."""
    n_feat = x.shape[2]
    x_np = x.numpy()
    y_np = y.numpy()

    results["minipfn nan-fill"][0.0] += model_task_accuracies(
        model, x, y, train_size, n_classes, device
    )
    lr_full = [
        _logreg_accuracy(
            x_np[b, :train_size],
            y_np[b, :train_size],
            x_np[b, train_size:],
            y_np[b, train_size:],
        )
        for b in range(x.shape[0])
    ]
    results["logreg impute"][0.0] += lr_full
    for alias in ("minipfn drop-context", "minipfn mean-impute"):
        results[alias][0.0] += results["minipfn nan-fill"][0.0][-x.shape[0] :]
    results["logreg retrain"][0.0] += lr_full

    for frac in drop_fracs:
        n_drop = min(n_feat - 1, max(1, round(n_feat * frac)))
        cols = torch.randperm(n_feat, generator=generator)[:n_drop]
        dropped = torch.zeros(n_feat, dtype=torch.bool)
        dropped[cols] = True

        x_nan = apply_test_missing(x, train_size, dropped.expand(x.shape[0], n_feat))
        results["minipfn nan-fill"][frac] += model_task_accuracies(
            model, x_nan, y, train_size, n_classes, device
        )

        x_dc = x[:, :, ~dropped]
        results["minipfn drop-context"][frac] += model_task_accuracies(
            model, x_dc, y, train_size, n_classes, device
        )

        means = _train_col_means(x, train_size)  # (B, F)
        x_imp = x.clone()
        fill = means.unsqueeze(1).expand_as(x_imp)
        test_dropped = torch.zeros_like(x_imp, dtype=torch.bool)
        test_dropped[:, train_size:, dropped] = True
        x_imp[test_dropped] = fill[test_dropped]
        results["minipfn mean-impute"][frac] += model_task_accuracies(
            model, x_imp, y, train_size, n_classes, device
        )

        keep_idx = (~dropped).nonzero().squeeze(1).numpy()
        x_nan_np = x_nan.numpy()
        for b in range(x.shape[0]):
            results["logreg impute"][frac].append(
                _logreg_accuracy(
                    x_np[b, :train_size],
                    y_np[b, :train_size],
                    x_nan_np[b, train_size:],
                    y_np[b, train_size:],
                )
            )
            results["logreg retrain"][frac].append(
                _logreg_accuracy(
                    x_np[b, :train_size][:, keep_idx],
                    y_np[b, :train_size],
                    x_np[b, train_size:][:, keep_idx],
                    y_np[b, train_size:],
                )
            )


def evaluate_synthetic(
    model: MiniPFN,
    prior_cfg: PriorConfig,
    device: torch.device,
    n_batches: int,
    batch_size: int,
    seed: int,
    drop_fracs: list[float],
) -> Results:
    generator = torch.Generator().manual_seed(seed)
    results: Results = defaultdict(lambda: defaultdict(list))
    for _ in range(n_batches):
        batch = sample_prior_batch(prior_cfg, batch_size, generator)
        evaluate_batch(
            model,
            batch.x,
            batch.y,
            batch.train_size,
            batch.n_classes,
            device,
            drop_fracs,
            generator,
            results,
        )
    return results


def evaluate_real(
    model: MiniPFN,
    device: torch.device,
    seed: int,
    drop_fracs: list[float],
    max_features: int,
    n_repeats: int = 5,
) -> dict[str, Results]:
    """Synthetic-to-real transfer check on small sklearn datasets."""
    all_results: dict[str, Results] = {}
    for name, loader in (("breast_cancer", load_breast_cancer), ("wine", load_wine)):
        data_x, data_y = loader(return_X_y=True)
        rng = np.random.default_rng(seed)
        row_order = rng.permutation(len(data_x))[:300]
        col_order = rng.permutation(data_x.shape[1])[:max_features]
        data_x = data_x[row_order][:, col_order]
        data_y = data_y[row_order]

        x = torch.tensor(data_x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(data_y, dtype=torch.long).unsqueeze(0)
        train_size = 100
        n_classes = torch.tensor([int(data_y.max()) + 1])

        results: Results = defaultdict(lambda: defaultdict(list))
        generator = torch.Generator().manual_seed(seed)
        for _ in range(n_repeats):
            evaluate_batch(
                model,
                x,
                y,
                train_size,
                n_classes,
                device,
                drop_fracs,
                generator,
                results,
            )
        all_results[name] = results
    return all_results


def print_table(
    title: str, results: Results, rates: list[float], label: str = "drop"
) -> None:
    all_rates = [0.0, *rates] if 0.0 not in rates else rates
    header = "strategy".ljust(24) + "".join(
        f"{label} {r:>4.0%}".rjust(14) for r in all_rates
    )
    print(f"\n## {title}\n\n{header}\n{'-' * len(header)}")
    order = [
        "minipfn native",
        "minipfn nan-fill",
        "minipfn drop-context",
        "minipfn mean-impute",
        "logreg mean-impute",
        "logreg mean+indicator",
        "logreg knn-impute",
        "logreg impute",
        "logreg retrain",
    ]
    ordered = [s for s in order if s in results] + [
        s for s in results if s not in order
    ]
    for strategy in ordered:
        by_rate = results[strategy]
        row = strategy.ljust(24)
        for rate in all_rates:
            accs = by_rate.get(rate, [])
            if accs:
                sem = float(np.std(accs) / math.sqrt(len(accs)))
                row += f"{np.mean(accs):.3f}±{sem:.3f}".rjust(14)
            else:
                row += "-".rjust(14)
        print(row)


def _run_cells(
    args: argparse.Namespace, model: MiniPFN, prior_cfg: PriorConfig
) -> None:
    device = resolve_device(args.device)
    miss_rates = [float(f) for f in args.miss_rates.split(",") if f]
    n_tasks = args.n_batches * args.batch_size
    mcar, mnar = evaluate_synthetic_cells(
        model,
        prior_cfg,
        device,
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
    if args.real:
        for name, results in evaluate_real_cells(
            model, device, args.seed, miss_rates, prior_cfg.max_features
        ).items():
            print_table(f"real data, per-row MCAR: {name}", results, miss_rates, "miss")


def _run_columns(
    args: argparse.Namespace, model: MiniPFN, prior_cfg: PriorConfig
) -> None:
    device = resolve_device(args.device)
    drop_fracs = [float(f) for f in args.drop_fracs.split(",") if f]
    synthetic = evaluate_synthetic(
        model, prior_cfg, device, args.n_batches, args.batch_size, args.seed, drop_fracs
    )
    print_table(
        f"synthetic tasks ({args.n_batches * args.batch_size} tasks)",
        synthetic,
        drop_fracs,
    )
    if args.real:
        for name, results in evaluate_real(
            model, device, args.seed, drop_fracs, prior_cfg.max_features
        ).items():
            print_table(f"real data: {name}", results, drop_fracs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/minipfn.pt")
    parser.add_argument(
        "--scenario", type=str, default="cells", choices=["cells", "columns"]
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--miss-rates", type=str, default="0.2,0.4,0.6")
    parser.add_argument("--mnar-rate", type=float, default=0.3)
    parser.add_argument("--drop-fracs", type=str, default="0.25,0.5")
    parser.add_argument("--real", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, resolve_device(args.device))
    prior_cfg = PriorConfig(**ckpt["prior_config"])
    if args.scenario == "cells":
        _run_cells(args, model, prior_cfg)
    else:
        _run_columns(args, model, prior_cfg)


if __name__ == "__main__":
    main()
