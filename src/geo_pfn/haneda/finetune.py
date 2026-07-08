"""Plan C: finetune TabPFN v2 on the Haneda folds: ``python -m geo_pfn.haneda.finetune``.

Wraps the official ``tabpfn.finetuning`` module with two project constraints:

- **License**: the installed ``FinetunedTabPFNRegressor`` hardcodes the
  v2.5 checkpoint, whose license forbids commercial use; only the v2 weights
  are safe for this project. ``make_finetuned_v2`` overrides the estimator
  factory to force ``ModelVersion.V2`` (the same passthrough upstream added
  in a later release).
- **Pairing**: folds, feature matrices, and metrics reuse ``geo_pfn.haneda``
  helpers with the same seed as the zero-shot run, so every finetuned fold is
  directly comparable with the committed ``results/haneda/imputation.json``
  arms (v2-reg @ LCSG, native and mean+ind).

Caveat recorded for the report: the wrapper's internal early-stopping split
(``validation_split_ratio``) is row-random, not borehole-grouped, so its
validation loss is mildly optimistic; the outer test folds stay leak-free.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from geo_pfn.haneda.data import (
    GROUP,
    TARGET,
    FeatureSet,
    Imputation,
    borehole_folds,
    load_haneda,
    prepare_fold,
)
from geo_pfn.haneda.runners import regression_metrics


def make_finetuned_v2(**kwargs: Any):
    """Build a FinetunedTabPFNRegressor that finetunes the v2 checkpoint.

    Imports lazily (the finetuning stack pulls in heavy optional deps) and
    subclasses at call time so the module stays importable without them.
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

    class _FinetunedV2(FinetunedTabPFNRegressor):
        def _create_estimator(self, config: dict[str, Any]) -> TabPFNRegressor:
            return TabPFNRegressor.create_default_for_version(
                version=ModelVersion.V2,
                **config,
                fit_mode="batched",
                differentiable_input=False,
            )

    return _FinetunedV2(**kwargs)


@dataclass(kw_only=True)
class FinetuneConfig:
    """One finetuning evaluation run over the Haneda folds."""

    data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv"
    out_dir: str = "results/haneda"
    device: str = "mps"
    n_folds: int = 5
    seed: int = 42
    feature_set: FeatureSet = FeatureSet.LCSG
    imputations: tuple[Imputation, ...] = (
        Imputation.NATIVE,
        Imputation.MEAN_INDICATOR,
    )
    epochs: int = 30
    learning_rate: float = 1e-5
    n_estimators_finetune: int = 2
    n_estimators_final_inference: int = 8
    folds_subset: tuple[int, ...] | None = None  # e.g. (0,) for a smoke run
    grouped_val: bool = True  # early-stopping val = held-out boreholes
    grouped_val_frac: float = 0.1

    def __post_init__(self) -> None:
        self.feature_set = FeatureSet(self.feature_set)
        self.imputations = tuple(Imputation(i) for i in self.imputations)
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.folds_subset is not None:
            bad = [f for f in self.folds_subset if not 0 <= f < self.n_folds]
            if bad:
                raise ValueError(f"folds_subset out of range: {bad}")
        if not 0.0 < self.grouped_val_frac < 0.5:
            raise ValueError("grouped_val_frac must be in (0, 0.5)")


def run(config: FinetuneConfig) -> None:
    t0 = time.monotonic()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_haneda(config.data_path)
    su = df[TARGET].to_numpy(dtype="float64")
    folds = borehole_folds(df[GROUP].to_numpy(), config.n_folds, config.seed)
    wanted = (
        set(config.folds_subset)
        if config.folds_subset is not None
        else set(range(config.n_folds))
    )

    records: list[dict] = []
    predictions: list[dict] = []
    for imputation in config.imputations:
        for fold, (train_idx, test_idx) in enumerate(folds):
            if fold not in wanted:
                continue
            x_train, x_test = prepare_fold(
                df, config.feature_set, imputation, train_idx, test_idx
            )
            su_train = su[train_idx]
            if config.grouped_val:
                # early-stopping validation on held-out boreholes, so it
                # measures the spatial generalization the test folds measure
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=config.grouped_val_frac,
                    random_state=config.seed,
                )
                rel_tr, rel_val = next(
                    splitter.split(su_train, groups=df[GROUP].to_numpy()[train_idx])
                )
                fit_args = (x_train[rel_tr], su_train[rel_tr])
                fit_kwargs = {"X_val": x_train[rel_val], "y_val": su_train[rel_val]}
            else:
                rel_tr = np.arange(len(train_idx))
                fit_args = (x_train, su_train)
                fit_kwargs = {}
            model = make_finetuned_v2(
                device=config.device,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                random_state=config.seed,
                n_finetune_ctx_plus_query_samples=len(rel_tr),
                n_estimators_finetune=config.n_estimators_finetune,
                n_estimators_final_inference=config.n_estimators_final_inference,
            )
            fit_start = time.monotonic()
            model.fit(*fit_args, **fit_kwargs)
            y_pred = model.predict(x_test)
            metrics = regression_metrics(su[test_idx], y_pred)
            records.append(
                {
                    "experiment": "finetune",
                    "task": "regression",
                    "model": "v2-reg-finetuned",
                    "feature_set": config.feature_set.value,
                    "imputation": imputation.value,
                    "context": None,
                    "val": "grouped" if config.grouped_val else "row-random",
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_finetune_train": len(rel_tr),
                    "n_test": len(test_idx),
                    "fit_seconds": round(time.monotonic() - fit_start, 1),
                    **metrics,
                }
            )
            predictions += [
                {
                    "experiment": "finetune",
                    "model": "v2-reg-finetuned",
                    "imputation": imputation.value,
                    "fold": fold,
                    "row": int(row),
                    "y_true": float(t),
                    "y_pred": float(p),
                }
                for row, t, p in zip(test_idx, su[test_idx], y_pred)
            ]
            _write_outputs(out_dir, config, records, predictions)
            print(
                f"[{time.monotonic() - t0:7.1f}s] finetune imp={imputation.value:9s} "
                f"fold={fold} rmse={metrics['rmse']:.3f} mae={metrics['mae']:.3f} "
                f"r2={metrics['r2']:.3f}",
                flush=True,
            )
    print(f"done in {time.monotonic() - t0:.1f}s -> {out_dir}", flush=True)


def _write_outputs(
    out_dir: Path,
    config: FinetuneConfig,
    records: list[dict],
    predictions: list[dict],
) -> None:
    payload = {
        "config": {
            **asdict(config),
            "feature_set": config.feature_set.value,
            "imputations": [i.value for i in config.imputations],
        },
        "records": records,
    }
    name = f"finetune-{'grouped' if config.grouped_val else 'rowrand'}"
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=1))
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    (pred_dir / f"{name}.json").write_text(json.dumps(predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = FinetuneConfig()
    parser.add_argument("--data-path", type=str, default=defaults.data_path)
    parser.add_argument("--out-dir", type=str, default=defaults.out_dir)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--n-folds", type=int, default=defaults.n_folds)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--feature-set", type=str, default=defaults.feature_set.value)
    parser.add_argument(
        "--imputations",
        type=str,
        default=",".join(i.value for i in defaults.imputations),
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--folds-subset", type=str, default="", help="e.g. '0' or '0,1' for smoke runs"
    )
    parser.add_argument(
        "--grouped-val",
        action=argparse.BooleanOptionalAction,
        default=defaults.grouped_val,
    )
    args = parser.parse_args()
    config = FinetuneConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
        device=args.device,
        n_folds=args.n_folds,
        seed=args.seed,
        feature_set=FeatureSet(args.feature_set),
        imputations=tuple(
            Imputation(t) for i in args.imputations.split(",") if (t := i.strip())
        ),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        grouped_val=args.grouped_val,
        folds_subset=tuple(
            int(t) for f in args.folds_subset.split(",") if (t := f.strip())
        )
        or None,
    )
    run(config)


if __name__ == "__main__":
    main()
