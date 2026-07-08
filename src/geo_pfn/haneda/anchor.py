"""Sparse-anchor cross-borehole transfer: ``python -m geo_pfn.haneda.anchor``.

The realistic prognosis question is *not* shallow->deep extrapolation within a
borehole (see ``infill.py``, which found that barely helps). It is: a target
borehole is sparsely sampled, but its few measured specimens — spread across
depth — pin down which of its densely-sampled neighbours it resembles, so the
unmeasured depths can borrow that neighbour's profile. Similarity travels
through the features already present (X, Y, depth, soil, grain size, cheap
geotech); the anchor rows resolve *which* neighbour to trust and *what level*
this borehole sits at.

Design: for each test borehole, ``k`` anchor rows are chosen stratified across
depth (one per depth-bin), the rest are query. Two arms predict the same query
rows:

- ``holdout`` — context = training boreholes only (no rows from the target);
- ``anchor``  — context = training boreholes + the k anchor rows of the test
  boreholes.

Because anchors are interspersed with the query in depth (unlike infill's
shallow->deep split), a query row is never far from an anchor. In-context
models (v2, TabICL) can attend to the specific matching neighbour; the retrain
baseline (hgbt) only gets k more rows in a global fit and cannot personalise
per target borehole — so ``anchor - holdout`` for ICL beyond hgbt is the
signature of genuine similarity-based transfer. Regression, native NaN, LCSG.
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

CORE_MODELS = ("v2-reg", "hgbt")


@dataclass(kw_only=True)
class AnchorConfig:
    """One sparse-anchor transfer run."""

    data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv"
    out_path: str = "results/haneda/anchor.json"
    device: str = "auto"
    feature_set: FeatureSet = FeatureSet.LCSG
    models: tuple[str, ...] = CORE_MODELS
    n_folds: int = 5
    seed: int = 42
    k_anchors: tuple[int, ...] = (1, 2, 3, 5)
    n_estimators: int | None = None

    def __post_init__(self) -> None:
        self.feature_set = FeatureSet(self.feature_set)
        self.models = tuple(self.models)
        self.k_anchors = tuple(int(k) for k in self.k_anchors)
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if any(k < 1 for k in self.k_anchors):
            raise ValueError("k_anchors must be >= 1")


def select_anchors(
    df, test_idx: np.ndarray, k: int, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Pick ``k`` depth-stratified anchor rows per test borehole; rest are query.

    A borehole must have at least ``2 * k`` rows to keep both a spread of anchors
    and a non-trivial query; smaller boreholes are dropped for this ``k``. Rows
    are binned into ``k`` equal depth intervals and one row is drawn per bin, so
    the anchors span the borehole's depth range. Returns global row indices.
    """
    depth = df["depth_m"].to_numpy()
    bores = df[GROUP].to_numpy()
    anchors: list[int] = []
    query: list[int] = []
    for bore in np.unique(bores[test_idx]):
        rows = test_idx[bores[test_idx] == bore]
        if len(rows) < 2 * k:
            continue
        order = rows[np.argsort(depth[rows])]
        bins = np.array_split(order, k)
        picked = [int(generator.choice(b)) for b in bins if len(b)]
        anchors.extend(picked)
        query.extend(r for r in order if r not in set(picked))
    return np.array(anchors, dtype=int), np.array(query, dtype=int)


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


def run(config: AnchorConfig) -> None:
    t0 = time.monotonic()
    df = load_haneda(config.data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    folds = borehole_folds(df[GROUP].to_numpy(), config.n_folds, config.seed)
    device = str(resolve_device(config.device))

    records: list[dict] = []
    for k in config.k_anchors:
        for fold, (train_idx, test_idx) in enumerate(folds):
            generator = np.random.default_rng(config.seed + 100 * k + fold)
            anchor_idx, query_idx = select_anchors(df, test_idx, k, generator)
            arms = (
                ("holdout", train_idx),
                ("anchor", np.concatenate([train_idx, anchor_idx])),
            )
            for arm, fit_idx in arms:
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
                            "experiment": "anchor",
                            "model": model,
                            "arm": arm,
                            "k_anchors": k,
                            "feature_set": config.feature_set.value,
                            "fold": fold,
                            "n_anchor": len(anchor_idx),
                            "n_query": len(query_idx),
                            **metrics,
                        }
                    )
                    print(
                        f"[{time.monotonic() - t0:7.1f}s] {model:8s} {arm:7s} k={k} "
                        f"fold={fold} rmse={metrics['rmse']:.3f} r2={metrics['r2']:.3f}",
                        flush=True,
                    )
                    _write(config, records)
    _summarize(records)
    print(f"done in {time.monotonic() - t0:.1f}s -> {config.out_path}", flush=True)


def _write(config: AnchorConfig, records: list[dict]) -> None:
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {**asdict(config), "feature_set": config.feature_set.value},
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=1))


def _summarize(records: list[dict]) -> None:
    """Print paired anchor - holdout RMSE per (model, k)."""
    print("\n## anchor - holdout (RMSE, paired per fold; negative = anchors help)")
    keys = dict.fromkeys((r["model"], r["k_anchors"]) for r in records)
    for model, k in keys:
        by_fold: dict[int, dict[str, float]] = {}
        for r in records:
            if r["model"] == model and r["k_anchors"] == k:
                by_fold.setdefault(r["fold"], {})[r["arm"]] = r["rmse"]
        diffs = [
            v["anchor"] - v["holdout"]
            for v in by_fold.values()
            if "anchor" in v and "holdout" in v
        ]
        hold = [v["holdout"] for v in by_fold.values() if "holdout" in v]
        anch = [v["anchor"] for v in by_fold.values() if "anchor" in v]
        d = np.array(diffs)
        sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(
            f"  {model:8s} k={k}: holdout={np.mean(hold):.3f} anchor={np.mean(anch):.3f} "
            f"diff={d.mean():+.3f} +- {sem:.3f} (n={len(d)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = AnchorConfig()
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
    parser.add_argument(
        "--k-anchors",
        type=str,
        default=",".join(str(k) for k in defaults.k_anchors),
    )
    parser.add_argument("--n-estimators", type=int, default=None)
    args = parser.parse_args()
    config = AnchorConfig(
        data_path=args.data_path,
        out_path=args.out,
        device=args.device,
        feature_set=FeatureSet(args.feature_set),
        models=tuple(t for m in args.models.split(",") if (t := m.strip())),
        n_folds=args.n_folds,
        seed=args.seed,
        k_anchors=tuple(int(t) for k in args.k_anchors.split(",") if (t := k.strip())),
        n_estimators=args.n_estimators,
    )
    run(config)


if __name__ == "__main__":
    main()
