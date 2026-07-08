"""Run the Haneda real-data evaluation: ``python -m geo_pfn.haneda.run``.

Experiments (docs/haneda-experiment-plan.md):

- ``ablation``   — feature-layer ablation (L/LC/LCS/LCSG/LSG), native NaN, all
  models: TabPFN v2 (regression + binned classification), mini-PFN checkpoints,
  HistGradientBoosting, linear, depth-only, dummy.
- ``imputation`` — missing-value strategies on the full feature set (LCSG).
- ``context``    — mini-PFN context-size scan on LCS.

All arms share the same borehole-grouped folds, the same train-fold quantile
bin edges, and (for the mini-PFN) the same per-fold context row draws, so
comparisons are paired. Aggregate metrics land in ``<out>/<experiment>.json``,
per-row predictions in ``<out>/predictions/<experiment>.csv`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geo_pfn.haneda.data import (
    GROUP,
    TARGET,
    FeatureSet,
    Imputation,
    borehole_folds,
    categorical_indices,
    load_haneda,
    prepare_fold,
    quantile_bin_labels,
)
from geo_pfn.haneda.runners import (
    ContextConfig,
    classification_metrics,
    make_baseline,
    make_tabpfn_v2,
    minipfn_predict_proba,
    regression_metrics,
)
from geo_pfn.minipfn.train import load_checkpoint, resolve_device

EXPERIMENTS = ("ablation", "imputation", "context")
ABLATION_SETS = (
    FeatureSet.L,
    FeatureSet.LC,
    FeatureSet.LCS,
    FeatureSet.LCSG,
    FeatureSet.LSG,
)
IMPUTATION_SET = FeatureSet.LCSG
CONTEXT_SET = FeatureSet.LCS


@dataclass(kw_only=True)
class RunConfig:
    """Configuration for one evaluation run."""

    data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv"
    out_dir: str = "results/haneda"
    device: str = "auto"
    n_folds: int = 5
    seed: int = 42
    n_bins: int = 4
    n_estimators: int | None = None
    experiments: tuple[str, ...] = EXPERIMENTS
    cells_checkpoint: str = "checkpoints/minipfn_cells.pt"
    vanilla_checkpoint: str = "checkpoints/minipfn_vanilla.pt"
    context_scan: str = "128:16,512:4,1024:2"
    save_predictions: bool = True
    skip_v2: bool = False

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if not 2 <= self.n_bins <= 4:
            raise ValueError("n_bins must be in [2, 4] (mini-PFN head is 4-way)")
        unknown = set(self.experiments) - set(EXPERIMENTS)
        if unknown:
            raise ValueError(f"unknown experiments: {sorted(unknown)}")
        parse_context_scan(self.context_scan)  # validate early


def parse_context_scan(spec: str) -> list[ContextConfig]:
    """Parse ``"128:16,512:4"`` into ContextConfigs (ctx_size:n_ensembles)."""
    configs = []
    for part in spec.split(","):
        if not part:
            continue
        ctx, _, ens = part.partition(":")
        configs.append(ContextConfig(ctx_size=int(ctx), n_ensembles=int(ens or 1)))
    if not configs:
        raise ValueError("context_scan must name at least one ctx_size:n_ensembles")
    return configs


def run(config: RunConfig) -> None:
    t0 = time.monotonic()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_haneda(config.data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    folds = borehole_folds(df[GROUP].to_numpy(), config.n_folds, config.seed)
    fold_bins = [
        (
            quantile_bin_labels(su[tr], su[tr], config.n_bins),
            quantile_bin_labels(su[tr], su[te], config.n_bins),
        )
        for tr, te in folds
    ]

    device = resolve_device(config.device)
    minis = {
        "mini-cells": load_checkpoint(config.cells_checkpoint, device)[0],
        "mini-vanilla": load_checkpoint(config.vanilla_checkpoint, device)[0],
    }
    default_ctx = ContextConfig()
    v2_cache: dict[str, object] = {}

    def get_v2(task: str):
        if task not in v2_cache:
            v2_cache[task] = make_tabpfn_v2(task, str(device), config.n_estimators)
        return v2_cache[task]

    records: list[dict] = []
    predictions: list[dict] = []

    def eval_arm(
        experiment: str,
        task: str,
        model: str,
        feature_set: FeatureSet,
        imputation: Imputation,
        ctx: ContextConfig | None = None,
    ) -> None:
        arm_metrics = []
        for fold, (train_idx, test_idx) in enumerate(folds):
            x_train, x_test = prepare_fold(
                df, feature_set, imputation, train_idx, test_idx
            )
            if model == "depth":  # depth_m is column 0 of every feature set
                x_train, x_test = x_train[:, :1], x_test[:, :1]
            if task == "classification":
                y_train, y_test = fold_bins[fold]
            else:
                y_train, y_test = su[train_idx], su[test_idx]

            if model in minis:
                proba = minipfn_predict_proba(
                    minis[model],
                    x_train,
                    y_train,
                    x_test,
                    config.n_bins,
                    ctx or default_ctx,
                    config.seed + 1000 * fold,
                    device,
                )
                y_pred: np.ndarray = proba.argmax(axis=1)
            elif model in ("v2-reg", "v2-clf"):
                est = get_v2(task)
                est.categorical_features_indices = (
                    categorical_indices(feature_set) or None
                )
                est.fit(x_train, y_train)
                y_pred = np.asarray(est.predict(x_test))
            else:
                est = make_baseline(model, task)
                est.fit(x_train, y_train)
                y_pred = np.asarray(est.predict(x_test))

            metrics = (
                classification_metrics(y_test, y_pred)
                if task == "classification"
                else regression_metrics(y_test, y_pred)
            )
            arm_metrics.append(metrics)
            records.append(
                {
                    "experiment": experiment,
                    "task": task,
                    "model": model,
                    "feature_set": feature_set.value,
                    "imputation": imputation.value,
                    "context": f"{ctx.ctx_size}x{ctx.n_ensembles}" if ctx else None,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **metrics,
                }
            )
            if config.save_predictions:
                for row, true, pred in zip(test_idx, y_test, y_pred):
                    predictions.append(
                        {
                            "experiment": experiment,
                            "task": task,
                            "model": model,
                            "feature_set": feature_set.value,
                            "imputation": imputation.value,
                            "context": f"{ctx.ctx_size}x{ctx.n_ensembles}"
                            if ctx
                            else None,
                            "fold": fold,
                            "row": int(row),
                            "su": float(su[row]),
                            "y_true": float(true),
                            "y_pred": float(pred),
                        }
                    )

        key = "accuracy" if task == "classification" else "rmse"
        mean = float(np.mean([m[key] for m in arm_metrics]))
        print(
            f"[{time.monotonic() - t0:7.1f}s] {experiment:10s} {task:14s} "
            f"{model:12s} fs={feature_set.value:4s} imp={imputation.value:9s} "
            f"ctx={ctx.ctx_size if ctx else '-'} {key}={mean:.3f}",
            flush=True,
        )
        _write_outputs(out_dir, config, records, predictions)

    v2_arms = [] if config.skip_v2 else ["v2"]

    if "ablation" in config.experiments:
        for fs in ABLATION_SETS:
            for v2 in v2_arms:
                eval_arm("ablation", "regression", f"{v2}-reg", fs, Imputation.NATIVE)
            for model in ("hgbt", "linear"):
                eval_arm("ablation", "regression", model, fs, Imputation.NATIVE)
            for v2 in v2_arms:
                eval_arm(
                    "ablation", "classification", f"{v2}-clf", fs, Imputation.NATIVE
                )
            for model in ("mini-cells", "mini-vanilla", "hgbt", "linear"):
                eval_arm("ablation", "classification", model, fs, Imputation.NATIVE)
        for model in ("depth", "dummy"):  # feature-set independent floors
            eval_arm("ablation", "regression", model, FeatureSet.L, Imputation.NATIVE)
            eval_arm(
                "ablation", "classification", model, FeatureSet.L, Imputation.NATIVE
            )

    if "imputation" in config.experiments:
        all_strategies = tuple(Imputation)
        arms: list[tuple[str, str, tuple[Imputation, ...]]] = [
            ("regression", "linear", all_strategies),
            ("regression", "hgbt", (Imputation.NATIVE, Imputation.MEAN)),
            (
                "classification",
                "mini-cells",
                tuple(s for s in all_strategies if s is not Imputation.MEAN_INDICATOR),
            ),
            ("classification", "linear", all_strategies),
            ("classification", "hgbt", (Imputation.NATIVE, Imputation.MEAN)),
        ]
        if not config.skip_v2:
            arms = [
                ("regression", "v2-reg", all_strategies),
                ("classification", "v2-clf", all_strategies),
                *arms,
            ]
        for task, model, strategies in arms:
            for strategy in strategies:
                eval_arm("imputation", task, model, IMPUTATION_SET, strategy)

    if "context" in config.experiments:
        for ctx in parse_context_scan(config.context_scan):
            for model in ("mini-cells", "mini-vanilla"):
                eval_arm(
                    "context",
                    "classification",
                    model,
                    CONTEXT_SET,
                    Imputation.NATIVE,
                    ctx=ctx,
                )

    _write_outputs(out_dir, config, records, predictions)
    print(f"done in {time.monotonic() - t0:.1f}s -> {out_dir}", flush=True)


def _write_outputs(
    out_dir: Path, config: RunConfig, records: list[dict], predictions: list[dict]
) -> None:
    for experiment in {r["experiment"] for r in records}:
        payload = {
            "config": asdict(config),
            "records": [r for r in records if r["experiment"] == experiment],
        }
        (out_dir / f"{experiment}.json").write_text(json.dumps(payload, indent=1))
    if predictions:
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        frame = pd.DataFrame(predictions)
        for experiment, group in frame.groupby("experiment"):
            group.to_csv(pred_dir / f"{experiment}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = RunConfig()
    parser.add_argument("--data-path", type=str, default=defaults.data_path)
    parser.add_argument("--out-dir", type=str, default=defaults.out_dir)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--n-folds", type=int, default=defaults.n_folds)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--n-bins", type=int, default=defaults.n_bins)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument(
        "--experiments", type=str, default=",".join(defaults.experiments)
    )
    parser.add_argument(
        "--cells-checkpoint", type=str, default=defaults.cells_checkpoint
    )
    parser.add_argument(
        "--vanilla-checkpoint", type=str, default=defaults.vanilla_checkpoint
    )
    parser.add_argument("--context-scan", type=str, default=defaults.context_scan)
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=defaults.save_predictions,
    )
    parser.add_argument(
        "--skip-v2", action=argparse.BooleanOptionalAction, default=defaults.skip_v2
    )
    args = parser.parse_args()
    config = RunConfig(
        **{
            **vars(args),
            "experiments": tuple(
                t for e in args.experiments.split(",") if (t := e.strip())
            ),
        }
    )
    run(config)


if __name__ == "__main__":
    main()
