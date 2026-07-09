"""Sparse-context borehole prediction: ``python -m geo_pfn.haneda.sparse_context``.

The core test of the borehole-similarity hypothesis, and the realistic deployment
regime: predict whole held-out boreholes from only a *few* other whole boreholes
as context (a sparse survey), not from all ~190. Similarity-awareness should
matter most exactly here — with few context holes the model must decide which of
them the target resembles and borrow that profile — so this is where geo-PFN's
two-stage row-metric could show the mechanism even though it is far smaller and
less trained than TabICL.

Two tests, both on real held-out boreholes (geology is measured there, so unlike
the virtual grid this is a genuine ablation):

- **A (sparsity sweep)** — feature set L (coords + depth). RMSE vs the number of
  context boreholes N, per model. The signal is the *relative* curve
  ``geopfn - tabicl`` as N shrinks.
- **B (cheap geology)** — at each N, does adding cheap large-scale geology
  (LSG = L + soil + grain size) to precious sparse boreholes lift local Su
  accuracy? Compares L vs LSG per model.

For each N a set of context boreholes is sampled from the training pool over
several seeds and averaged. Run in the overlay env (TabICL)::

    uv run --with tabicl python -m geo_pfn.haneda.sparse_context --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from geo_pfn.geopfn.eval_anchor import load_geopfn
from geo_pfn.geopfn.predict import CoherentConfig, predict_geopfn_coherent
from geo_pfn.haneda.data import (
    GROUP,
    TARGET,
    FeatureSet,
    borehole_folds,
    feature_columns,
    load_haneda,
)


@dataclass(kw_only=True)
class SparseConfig:
    data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv"
    out_path: str = "results/haneda/sparse_context.json"
    checkpoint: str = "checkpoints/geopfn2stage.pt"
    device: str = "cpu"
    feature_sets: tuple[str, ...] = ("L", "LSG")
    models: tuple[str, ...] = ("tabicl", "geopfn", "hgbt")
    context_sizes: tuple[int, ...] = (3, 6, 12, 25, 50, 100, -1)  # -1 = all pool holes
    n_seeds: int = 4
    query_fold: int = 0
    n_folds: int = 5
    seed: int = 42
    coherent: dict = field(
        default_factory=lambda: {
            "n_holes": 8,
            "n_candidates": 24,
            "n_ensembles": 8,
        }
    )

    def __post_init__(self) -> None:
        self.feature_sets = tuple(self.feature_sets)
        self.models = tuple(self.models)
        self.context_sizes = tuple(int(n) for n in self.context_sizes)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _predict(
    model: str,
    fs: FeatureSet,
    df,
    ctx_rows: np.ndarray,
    query_rows: np.ndarray,
    query_bores: np.ndarray,
    su: np.ndarray,
    bores: np.ndarray,
    geopfn,
    coherent: CoherentConfig,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Predict Su for the query rows with one model and feature set."""
    x = df[feature_columns(fs)].to_numpy(dtype=np.float64)
    xc, xq = x[ctx_rows], x[query_rows]
    yc = su[ctx_rows]

    if model == "geopfn":
        pred = np.full(len(query_rows), np.nan)
        for b in np.unique(query_bores):
            q = query_bores == b
            pred[q] = predict_geopfn_coherent(
                geopfn, xc, yc, bores[ctx_rows], xq[q], coherent, seed + int(b), device
            )
        return pred
    if model == "tabicl":
        from tabicl import TabICLRegressor

        est = TabICLRegressor(
            device=None if device.type == "auto" else device.type, random_state=42
        )
    else:
        from geo_pfn.haneda.runners import make_baseline

        est = make_baseline(model, "regression")
    est.fit(xc, yc)
    return np.asarray(est.predict(xq))


def run(config: SparseConfig) -> None:
    t0 = time.monotonic()
    df = load_haneda(config.data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    bores = df[GROUP].to_numpy()
    device = torch.device(config.device)

    folds = borehole_folds(bores, config.n_folds, config.seed)
    _, query_rows = folds[config.query_fold]
    query_rows = np.asarray(query_rows)
    query_bores = bores[query_rows]
    pool_bores = np.array(sorted(set(bores) - set(query_bores)))
    print(
        f"query fold {config.query_fold}: {len(np.unique(query_bores))} boreholes "
        f"({len(query_rows)} rows); pool = {len(pool_bores)} boreholes",
        flush=True,
    )

    geopfn = (
        load_geopfn(config.checkpoint, device) if "geopfn" in config.models else None
    )
    coherent = CoherentConfig(**config.coherent)

    records: list[dict] = []
    for n_ctx in config.context_sizes:
        n_eff = len(pool_bores) if n_ctx < 0 else min(n_ctx, len(pool_bores))
        for s in range(config.n_seeds):
            gen = np.random.default_rng(config.seed + 1000 * (n_ctx % 997) + s)
            pick = (
                pool_bores
                if n_ctx < 0
                else gen.choice(pool_bores, n_eff, replace=False)
            )
            ctx_rows = np.where(np.isin(bores, pick))[0]
            for fs_name in config.feature_sets:
                fs = FeatureSet(fs_name)
                for model in config.models:
                    pred = _predict(
                        model,
                        fs,
                        df,
                        ctx_rows,
                        query_rows,
                        query_bores,
                        su,
                        bores,
                        geopfn,
                        coherent,
                        device,
                        config.seed + s,
                    )
                    ok = ~np.isnan(pred)
                    rmse = _rmse(su[query_rows][ok], pred[ok])
                    records.append(
                        {
                            "model": model,
                            "feature_set": fs_name,
                            "n_ctx": n_eff,
                            "n_ctx_req": n_ctx,
                            "seed": s,
                            "n_query": int(ok.sum()),
                            "rmse": rmse,
                        }
                    )
                    print(
                        f"[{time.monotonic() - t0:6.1f}s] N={n_eff:>3} s={s} "
                        f"{model:7s} {fs_name:4s} rmse={rmse:6.2f}",
                        flush=True,
                    )
            if n_ctx < 0:
                break  # "all" is deterministic; one seed suffices
        _write(config, records)
    _summarize(records)
    print(f"done in {time.monotonic() - t0:.1f}s -> {config.out_path}", flush=True)


def _write(config: SparseConfig, records: list[dict]) -> None:
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": asdict(config), "records": records}, indent=1))


def _summarize(records: list[dict]) -> None:
    """Print mean RMSE per (feature_set, N, model) and the geopfn - tabicl gap."""
    fss = dict.fromkeys(r["feature_set"] for r in records)
    ns = sorted(dict.fromkeys(r["n_ctx"] for r in records))
    models = dict.fromkeys(r["model"] for r in records)
    for fs in fss:
        print(f"\n## feature set {fs}: mean RMSE by context size")
        header = "  N     " + "".join(f"{m:>9s}" for m in models)
        if "geopfn" in models and "tabicl" in models:
            header += "   gp-tab"
        print(header)
        for n in ns:
            cells, means = "", {}
            for m in models:
                vals = [
                    r["rmse"]
                    for r in records
                    if r["feature_set"] == fs and r["n_ctx"] == n and r["model"] == m
                ]
                means[m] = float(np.mean(vals)) if vals else float("nan")
                cells += f"{means[m]:9.2f}"
            gap = ""
            if "geopfn" in means and "tabicl" in means:
                gap = f"   {means['geopfn'] - means['tabicl']:+7.2f}"
            print(f"  {n:>4}  {cells}{gap}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    d = SparseConfig()
    parser.add_argument("--data-path", type=str, default=d.data_path)
    parser.add_argument("--out", type=str, default=d.out_path)
    parser.add_argument("--checkpoint", type=str, default=d.checkpoint)
    parser.add_argument("--device", type=str, default=d.device)
    parser.add_argument("--feature-sets", type=str, default=",".join(d.feature_sets))
    parser.add_argument("--models", type=str, default=",".join(d.models))
    parser.add_argument(
        "--context-sizes", type=str, default=",".join(str(n) for n in d.context_sizes)
    )
    parser.add_argument("--n-seeds", type=int, default=d.n_seeds)
    parser.add_argument("--query-fold", type=int, default=d.query_fold)
    args = parser.parse_args()
    config = SparseConfig(
        data_path=args.data_path,
        out_path=args.out,
        checkpoint=args.checkpoint,
        device=args.device,
        feature_sets=tuple(t for f in args.feature_sets.split(",") if (t := f.strip())),
        models=tuple(t for m in args.models.split(",") if (t := m.strip())),
        context_sizes=tuple(
            int(t) for n in args.context_sizes.split(",") if (t := n.strip())
        ),
        n_seeds=args.n_seeds,
        query_fold=args.query_fold,
    )
    run(config)


if __name__ == "__main__":
    main()
