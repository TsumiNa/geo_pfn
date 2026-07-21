"""Borehole infill experiment: ``python -m geo_pfn.haneda.infill``.

The whole-borehole holdout protocol (``geo_pfn.haneda.run``) leaves ~1/5 of the
best model's residual as a spatially-incoherent per-borehole bias (ICC 0.195),
worth ~1.9 RMSE — unrecoverable for a brand-new borehole, but recoverable when
the borehole's *shallow* specimens are already measured and only the *deep*
part is predicted (the real prognosis workflow: drill and test the top, infer
deeper).

This module measures that recoverable gain with a fold-paired design. For each
test borehole, its deepest ``query_frac`` rows are the fixed query set. Two arms
predict the same rows:

- ``holdout`` — fit on the training boreholes only (context has nothing from the
  query borehole);
- ``infill``  — fit on the training boreholes plus the shallow rows of the test
  boreholes.

The per-fold RMSE difference (infill - holdout) is the value of same-borehole
shallow context. In-context models (v2, TabICL) get those rows as context; the
retrain baselines (hgbt, linear) get them added to their training set, so the
comparison isolates whether in-context learning exploits the context better than
simply enlarging the training set. Regression only, native NaN, LCSG.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
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
)
from geo_pfn.haneda.runners import make_baseline, make_tabpfn_v2, regression_metrics
from geo_pfn.minipfn.train import resolve_device

CORE_MODELS = ("v2-reg", "hgbt", "linear")
MIN_BOREHOLE_ROWS = 4  # need at least a couple shallow + a couple deep rows


@dataclass(kw_only=True)
class InfillConfig:
    """One borehole-infill evaluation run."""

    data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv"
    out_path: str = "results/haneda/infill.json"
    device: str = "auto"
    feature_set: FeatureSet = FeatureSet.LCSG
    models: tuple[str, ...] = CORE_MODELS
    n_folds: int = 5
    seed: int = 42
    query_frac: float = 0.5
    n_estimators: int | None = None

    def __post_init__(self) -> None:
        self.feature_set = FeatureSet(self.feature_set)
        self.models = tuple(self.models)
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if not 0.0 < self.query_frac < 1.0:
            raise ValueError("query_frac must be in (0, 1)")


def depth_split(
    df, test_idx: np.ndarray, query_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split test-borehole rows into (shallow context, deep query) by depth.

    Within each borehole with at least ``MIN_BOREHOLE_ROWS`` rows, the deepest
    ``query_frac`` (most negative ``depth_m``) become query, the rest shallow
    context. Smaller boreholes are dropped (they cannot supply both). Returns
    global row indices.
    """
    depth = df["depth_m"].to_numpy()
    bores = df[GROUP].to_numpy()
    shallow: list[int] = []
    query: list[int] = []
    for bore in np.unique(bores[test_idx]):
        rows = test_idx[bores[test_idx] == bore]
        if len(rows) < MIN_BOREHOLE_ROWS:
            continue
        order = rows[np.argsort(depth[rows])]  # deepest (most negative) first
        n_query = max(1, min(len(rows) - 1, round(len(rows) * query_frac)))
        query.extend(order[:n_query])
        shallow.extend(order[n_query:])
    return np.array(shallow, dtype=int), np.array(query, dtype=int)


def _predict(
    model: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_query: np.ndarray,
    device: str,
    n_estimators: int | None,
) -> np.ndarray:
    if model == "v2-reg":
        est = make_tabpfn_v2("regression", device, n_estimators)
    elif model == "tabicl":
        from tabicl import TabICLRegressor  # ephemeral overlay dependency

        est = TabICLRegressor(device=None if device == "auto" else device)
    else:
        est = make_baseline(model, "regression")
    est.fit(x_fit, y_fit)
    return np.asarray(est.predict(x_query))


def run(config: InfillConfig) -> None:
    t0 = time.monotonic()
    df = load_haneda(config.data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    folds = borehole_folds(df[GROUP].to_numpy(), config.n_folds, config.seed)
    device = str(resolve_device(config.device))

    records: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        shallow_idx, query_idx = depth_split(df, test_idx, config.query_frac)
        holdout_fit = train_idx
        infill_fit = np.concatenate([train_idx, shallow_idx])
        for arm, fit_idx in (("holdout", holdout_fit), ("infill", infill_fit)):
            x_fit, x_query = prepare_fold(
                df, config.feature_set, Imputation.NATIVE, fit_idx, query_idx
            )
            for model in config.models:
                pred = _predict(
                    model, x_fit, su[fit_idx], x_query, device, config.n_estimators
                )
                metrics = regression_metrics(su[query_idx], pred)
                records.append(
                    {
                        "experiment": "infill",
                        "model": model,
                        "arm": arm,
                        "feature_set": config.feature_set.value,
                        "fold": fold,
                        "n_fit": len(fit_idx),
                        "n_shallow_ctx": len(shallow_idx),
                        "n_query": len(query_idx),
                        **metrics,
                    }
                )
                print(
                    f"[{time.monotonic() - t0:7.1f}s] {model:8s} {arm:7s} fold={fold} "
                    f"rmse={metrics['rmse']:.3f} r2={metrics['r2']:.3f}",
                    flush=True,
                )
                _write(config, records)
    _summarize(records)
    print(f"done in {time.monotonic() - t0:.1f}s -> {config.out_path}", flush=True)


def _write(config: InfillConfig, records: list[dict]) -> None:
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {**asdict(config), "feature_set": config.feature_set.value},
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=1))


def _summarize(records: list[dict]) -> None:
    """Print paired infill - holdout RMSE per model (mean +- sem over folds)."""
    models = dict.fromkeys(r["model"] for r in records)
    print("\n## infill - holdout (RMSE, paired per fold; negative = infill helps)")
    for model in models:
        by_fold: dict[int, dict[str, float]] = {}
        for r in records:
            if r["model"] == model:
                by_fold.setdefault(r["fold"], {})[r["arm"]] = r["rmse"]
        diffs = [
            v["infill"] - v["holdout"]
            for v in by_fold.values()
            if "infill" in v and "holdout" in v
        ]
        hold = [v["holdout"] for v in by_fold.values() if "holdout" in v]
        infl = [v["infill"] for v in by_fold.values() if "infill" in v]
        d = np.array(diffs)
        sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(
            f"  {model:8s} holdout={np.mean(hold):.3f} infill={np.mean(infl):.3f} "
            f"diff={d.mean():+.3f} +- {sem:.3f} (n={len(d)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = InfillConfig()
    parser.add_argument("--data-path", type=str, default=defaults.data_path)
    parser.add_argument("--out", type=str, default=defaults.out_path)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--feature-set", type=str, default=defaults.feature_set.value)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(defaults.models),
        help="comma-separated: v2-reg, hgbt, linear, tabicl",
    )
    parser.add_argument("--n-folds", type=int, default=defaults.n_folds)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--query-frac", type=float, default=defaults.query_frac)
    parser.add_argument("--n-estimators", type=int, default=None)
    args = parser.parse_args()
    config = InfillConfig(
        data_path=args.data_path,
        out_path=args.out,
        device=args.device,
        feature_set=FeatureSet(args.feature_set),
        models=tuple(t for m in args.models.split(",") if (t := m.strip())),
        n_folds=args.n_folds,
        seed=args.seed,
        query_frac=args.query_frac,
        n_estimators=args.n_estimators,
    )
    run(config)


if __name__ == "__main__":
    main()
