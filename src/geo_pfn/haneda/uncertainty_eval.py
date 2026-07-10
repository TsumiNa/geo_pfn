"""Sparse-regime uncertainty calibration: ``python -m geo_pfn.haneda.uncertainty_eval``.

The niche question: when context is only a few boreholes, is geo-PFN's predicted
uncertainty *trustworthy*? For each context size N (same protocol as
``sparse_context``: fixed query fold, N whole boreholes sampled as context,
several seeds), every query row gets a full mixture-of-histograms prediction
(:func:`predict_geopfn_coherent_dist`) and we score, per (feature_set, N):

- ``cov@c``    empirical coverage of the central c-interval (calibrated = c);
               below nominal = over-confident, the failure mode that matters
- ``w90``      mean width of the 90% interval (sharpness; only meaningful when
               calibrated)
- ``crps``     mean CRPS (calibration + sharpness in one proper score)
- ``nll``      mean negative log density at the truth (floored); ``outside`` is
               the fraction of truths with zero density under every component —
               catastrophic over-confidence
- ``std_corr`` Spearman corr(predicted std, |error|) — does the model know when
               it is wrong?
- ``rmse``     point RMSE (sanity; matches the sparse_context numbers)

Run: ``uv run python -m geo_pfn.haneda.uncertainty_eval --checkpoint <ckpt>``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from geo_pfn.geopfn.calibration import (
    mixture_crps,
    mixture_nll,
    mixture_quantiles,
    mixture_stats,
)
from geo_pfn.geopfn.eval_anchor import load_geopfn
from geo_pfn.geopfn.predict import CoherentConfig, predict_geopfn_coherent_dist
from geo_pfn.util import resolve_device
from geo_pfn.haneda.data import (
    GROUP,
    TARGET,
    FeatureSet,
    borehole_folds,
    feature_columns,
    load_haneda,
)

COVERAGES = (0.5, 0.8, 0.9, 0.95)


@dataclass(kw_only=True)
class UncertaintyConfig:
    data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv"
    out_path: str = "results/haneda/uncertainty_sparse.json"
    checkpoint: str = "checkpoints/geopfn2stage.pt"
    device: str = "cpu"
    feature_sets: tuple[str, ...] = ("L", "LCSG")
    context_sizes: tuple[int, ...] = (3, 6, 12, 25, 50, 100, -1)
    n_seeds: int = 4
    query_fold: int = 0
    n_folds: int = 5
    seed: int = 42
    coherent: dict = field(
        default_factory=lambda: {"n_holes": 8, "n_candidates": 24, "n_ensembles": 8}
    )

    def __post_init__(self) -> None:
        self.feature_sets = tuple(self.feature_sets)
        self.context_sizes = tuple(int(n) for n in self.context_sizes)


def _quantile_levels() -> np.ndarray:
    levels: list[float] = []
    for c in COVERAGES:
        levels += [(1 - c) / 2, (1 + c) / 2]
    return np.array(levels)


def run(config: UncertaintyConfig) -> None:
    t0 = time.monotonic()
    df = load_haneda(config.data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    bores = df[GROUP].to_numpy()
    device = resolve_device(config.device)

    folds = borehole_folds(bores, config.n_folds, config.seed)
    _, query_rows = folds[config.query_fold]
    query_rows = np.asarray(query_rows)
    query_bores = bores[query_rows]
    pool_bores = np.array(sorted(set(bores) - set(query_bores)))
    print(
        f"query fold {config.query_fold}: {len(np.unique(query_bores))} boreholes "
        f"({len(query_rows)} rows); pool = {len(pool_bores)}",
        flush=True,
    )

    model = load_geopfn(config.checkpoint, device)
    coherent = CoherentConfig(**config.coherent)
    levels = _quantile_levels()

    records: list[dict] = []
    for fs_name in config.feature_sets:
        fs = FeatureSet(fs_name)
        x = df[feature_columns(fs)].to_numpy(dtype=np.float64)
        for n_ctx in config.context_sizes:
            n_eff = len(pool_bores) if n_ctx < 0 else min(n_ctx, len(pool_bores))
            acc: dict[str, list[np.ndarray]] = {
                k: [] for k in ("crps", "nll", "outside", "std", "err", "w90", "qhits")
            }
            n_seeds = 1 if n_ctx < 0 else config.n_seeds
            for s in range(n_seeds):
                gen = np.random.default_rng(config.seed + 1000 * (n_ctx % 997) + s)
                pick = (
                    pool_bores
                    if n_ctx < 0
                    else gen.choice(pool_bores, n_eff, replace=False)
                )
                ctx_rows = np.where(np.isin(bores, pick))[0]
                for b in np.unique(query_bores):
                    q = query_rows[query_bores == b]
                    dist = predict_geopfn_coherent_dist(
                        model,
                        x[ctx_rows],
                        su[ctx_rows],
                        bores[ctx_rows],
                        x[q],
                        coherent,
                        config.seed + s + int(b),
                        device,
                    )
                    y = su[q]
                    args = (dist.probs, dist.ctx_mean, dist.ctx_std, dist.z_edges)
                    mean, std = mixture_stats(*args)
                    qs = mixture_quantiles(*args, levels)  # (Q, 2*len(COVERAGES))
                    acc["crps"].append(mixture_crps(*args, y))
                    nll, outside = mixture_nll(*args, y)
                    acc["nll"].append(nll)
                    acc["outside"].append(outside.astype(float))
                    acc["std"].append(std)
                    acc["err"].append(np.abs(y - mean))
                    hits = np.stack(
                        [
                            (y >= qs[:, 2 * i]) & (y <= qs[:, 2 * i + 1])
                            for i in range(len(COVERAGES))
                        ],
                        axis=1,
                    )
                    acc["qhits"].append(hits.astype(float))
                    i90 = COVERAGES.index(0.9)
                    acc["w90"].append(qs[:, 2 * i90 + 1] - qs[:, 2 * i90])
            cat = {k: np.concatenate(v) for k, v in acc.items()}
            rho = spearmanr(cat["std"], cat["err"]).statistic
            rec = {
                "feature_set": fs_name,
                "n_ctx": n_eff,
                "n_rows": int(len(cat["crps"])),
                "rmse": round(float(np.sqrt((cat["err"] ** 2).mean())), 3),
                "crps": round(float(cat["crps"].mean()), 3),
                "nll": round(float(cat["nll"].mean()), 3),
                "outside_frac": round(float(cat["outside"].mean()), 4),
                "w90": round(float(cat["w90"].mean()), 2),
                "std_corr": round(float(rho), 3),
                **{
                    f"cov@{c}": round(float(cat["qhits"][:, i].mean()), 4)
                    for i, c in enumerate(COVERAGES)
                },
            }
            records.append(rec)
            print(
                f"[{time.monotonic() - t0:7.1f}s] {fs_name:5s} N={n_eff:>3} "
                f"crps={rec['crps']:6.2f} nll={rec['nll']:5.2f} "
                f"out={rec['outside_frac']:.3f} w90={rec['w90']:6.1f} "
                f"cov90={rec['cov@0.9']:.3f} rho={rec['std_corr']:+.2f} "
                f"rmse={rec['rmse']:.2f}",
                flush=True,
            )
            _write(config, records)
    _summarize(records)
    print(f"done in {time.monotonic() - t0:.1f}s -> {config.out_path}", flush=True)


def _write(config: UncertaintyConfig, records: list[dict]) -> None:
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": asdict(config), "records": records}, indent=1))


def _summarize(records: list[dict]) -> None:
    for fs in dict.fromkeys(r["feature_set"] for r in records):
        print(f"\n## {fs}: calibration by context size (nominal cov = column header)")
        head = f"  {'N':>4} {'crps':>7} {'nll':>6} {'out%':>6} {'w90':>7} "
        head += " ".join(f"cov@{c:<4}" for c in COVERAGES) + "  std_rho   rmse"
        print(head)
        for r in records:
            if r["feature_set"] != fs:
                continue
            row = (
                f"  {r['n_ctx']:>4} {r['crps']:7.2f} {r['nll']:6.2f} "
                f"{100 * r['outside_frac']:6.2f} {r['w90']:7.1f} "
            )
            row += " ".join(f"{r[f'cov@{c}']:7.3f}" for c in COVERAGES)
            row += f"  {r['std_corr']:+7.3f} {r['rmse']:6.2f}"
            print(row)


def main() -> None:
    d = UncertaintyConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=str, default=d.data_path)
    p.add_argument("--out", type=str, default=d.out_path)
    p.add_argument("--checkpoint", type=str, default=d.checkpoint)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--feature-sets", type=str, default=",".join(d.feature_sets))
    p.add_argument(
        "--context-sizes",
        type=str,
        default=",".join(str(n) for n in d.context_sizes),
    )
    p.add_argument("--n-seeds", type=int, default=d.n_seeds)
    a = p.parse_args()
    run(
        UncertaintyConfig(
            data_path=a.data_path,
            out_path=a.out,
            checkpoint=a.checkpoint,
            device=a.device,
            feature_sets=tuple(
                t for f in a.feature_sets.split(",") if (t := f.strip())
            ),
            context_sizes=tuple(
                int(t) for n in a.context_sizes.split(",") if (t := n.strip())
            ),
            n_seeds=a.n_seeds,
        )
    )


if __name__ == "__main__":
    main()
