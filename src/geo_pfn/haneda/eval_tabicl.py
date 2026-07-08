"""Paired TabICLv2 baseline on the Haneda folds.

TabICL is not a project dependency; run in an ephemeral overlay environment:

    uv run --with tabicl python -m geo_pfn.haneda.eval_tabicl

Evaluates TabICLRegressor (raw Su) and TabICLClassifier (train-fold quantile
bins) on the same borehole-grouped folds and feature matrices as the TabPFN v2
arms in ``geo_pfn.haneda.run``, so results in ``results/haneda/tabicl.json``
are fold-paired with ``results/haneda/imputation.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from geo_pfn.haneda.data import (
    GROUP,
    TARGET,
    FeatureSet,
    Imputation,
    borehole_folds,
    load_haneda,
    prepare_fold,
    quantile_bin_labels,
)
from geo_pfn.haneda.runners import classification_metrics, regression_metrics


def run(
    data_path: str,
    out_path: str,
    device: str | None,
    feature_set: FeatureSet,
    imputations: tuple[Imputation, ...],
    n_folds: int,
    seed: int,
    n_bins: int,
) -> None:
    from tabicl import TabICLClassifier, TabICLRegressor  # ephemeral dependency

    df = load_haneda(data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    folds = borehole_folds(df[GROUP].to_numpy(), n_folds, seed)

    results = []
    t0 = time.monotonic()
    for imputation in imputations:
        for fold, (train_idx, test_idx) in enumerate(folds):
            x_train, x_test = prepare_fold(
                df, feature_set, imputation, train_idx, test_idx
            )

            reg = TabICLRegressor(device=device, random_state=seed)
            reg.fit(x_train, su[train_idx])
            metrics = regression_metrics(su[test_idx], np.asarray(reg.predict(x_test)))
            results.append(
                {
                    "task": "regression",
                    "model": "tabicl",
                    "feature_set": feature_set.value,
                    "imputation": imputation.value,
                    "fold": fold,
                    **metrics,
                }
            )
            print(
                f"[{time.monotonic() - t0:7.1f}s] tabicl-reg {imputation.value:9s} "
                f"fold={fold} rmse={metrics['rmse']:.3f} r2={metrics['r2']:.3f}",
                flush=True,
            )

            y_train = quantile_bin_labels(su[train_idx], su[train_idx], n_bins)
            y_test = quantile_bin_labels(su[train_idx], su[test_idx], n_bins)
            clf = TabICLClassifier(device=device, random_state=seed)
            clf.fit(x_train, y_train)
            metrics = classification_metrics(y_test, np.asarray(clf.predict(x_test)))
            results.append(
                {
                    "task": "classification",
                    "model": "tabicl",
                    "feature_set": feature_set.value,
                    "imputation": imputation.value,
                    "fold": fold,
                    **metrics,
                }
            )
            print(
                f"[{time.monotonic() - t0:7.1f}s] tabicl-clf {imputation.value:9s} "
                f"fold={fold} acc={metrics['accuracy']:.3f} "
                f"f1={metrics['macro_f1']:.3f}",
                flush=True,
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"saved -> {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=str, default="data/pilot_Su_domain_block_mod4Liu.csv"
    )
    parser.add_argument("--out", type=str, default="results/haneda/tabicl.json")
    parser.add_argument(
        "--device", type=str, default="cpu", help="'cpu', 'mps', or 'cuda'"
    )
    parser.add_argument("--feature-set", type=str, default=FeatureSet.LCSG.value)
    parser.add_argument("--imputations", type=str, default="native,mean+ind")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=4)
    args = parser.parse_args()
    run(
        data_path=args.data_path,
        out_path=args.out,
        device=args.device,
        feature_set=FeatureSet(args.feature_set),
        imputations=tuple(
            Imputation(t) for i in args.imputations.split(",") if (t := i.strip())
        ),
        n_folds=args.n_folds,
        seed=args.seed,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
