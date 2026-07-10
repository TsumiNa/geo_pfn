"""Fit the geo-realistic prior's empirical constants from the pilot CSV.

Run once to (re)generate ``haneda_stats.json`` next to this module::

    uv run python -m geo_pfn.geoprior.fit_stats

The realistic prior (``realistic.py``) reads that JSON at import time so training
never touches the CSV. Everything fitted here is a *marginal* or *second-order*
summary — quantile grids, a correlation matrix, PC1 loadings, and a single
coupling coefficient — so the synthetic tables reproduce the real cheap-feature
distribution's shape and its weak, redundant link to Su, not any real row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geo_pfn.haneda.data import CHEAP_COLUMNS, GRAIN_COLUMNS, load_haneda

STATS_PATH = Path(__file__).with_name("haneda_stats.json")
_N_QUANTILES = 41  # 0, 2.5, ..., 100 percentiles for marginal mapping


def _quantile_grid(values: np.ndarray) -> list[float]:
    qs = np.linspace(0.0, 1.0, _N_QUANTILES)
    return [round(float(v), 6) for v in np.quantile(values, qs)]


def fit(data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv") -> dict:
    df = load_haneda(data_path)
    cheap = list(CHEAP_COLUMNS)
    grain = list(GRAIN_COLUMNS)

    # --- cheap-feature cluster: marginals, correlation, principal axis ---
    present = df[cheap].notna().all(axis=1).to_numpy()
    block = df.loc[present, cheap].to_numpy(dtype=np.float64)
    standardized = (block - block.mean(0)) / block.std(0)
    corr = np.corrcoef(standardized, rowvar=False)
    _, _, vt = np.linalg.svd(standardized - standardized.mean(0), full_matrices=False)
    singular = np.linalg.svd(standardized - standardized.mean(0), compute_uv=False)
    ev = (singular**2 / (singular**2).sum()).tolist()
    pc1 = vt[0]
    # orient PC1 so that Wn loads positive (the "wetter/softer" direction)
    if pc1[cheap.index("Wn")] < 0:
        pc1 = -pc1

    # --- Su coupling: how much the softness axis adds beyond depth ---
    log_su = np.log(df.Su.to_numpy(dtype=np.float64))
    depth = df.depth_m.to_numpy(dtype=np.float64)
    soft_score = standardized @ pc1  # PC1 score on present rows
    lz, dz = log_su[present], depth[present]

    def _ols_r2(x_cols: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        x = np.column_stack([np.ones(len(y)), x_cols])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        r2 = 1.0 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
        return beta, r2

    _, depth_r2 = _ols_r2(dz[:, None], lz)
    beta_full, depth_soft_r2 = _ols_r2(np.column_stack([dz, soft_score]), lz)
    # standardized coupling: d(logSu) per 1 SD of softness, holding depth
    soft_beta_std = float(beta_full[2] * soft_score.std())

    # --- grain-size block: zero-inflation + positive-part marginals ---
    grain_stats = {}
    for col in grain:
        s = df[col].dropna().to_numpy(dtype=np.float64)
        pos = s[s > 0]
        grain_stats[col] = {
            "zero_frac": round(float((s == 0).mean()), 4),
            "pos_quantiles": _quantile_grid(pos) if len(pos) else [0.0] * _N_QUANTILES,
        }

    return {
        "_source": Path(data_path).name,
        "_n_rows": int(len(df)),
        "cheap_columns": cheap,
        "cheap_quantiles": {
            c: _quantile_grid(df[c].dropna().to_numpy()) for c in cheap
        },
        "cheap_corr": np.round(corr, 4).tolist(),
        "cheap_pc1": np.round(pc1, 4).tolist(),
        "cheap_explained_var": [round(v, 4) for v in ev],
        "grain_columns": grain,
        "grain": grain_stats,
        "coupling": {
            "depth_r2": round(float(depth_r2), 4),
            "depth_soft_r2": round(float(depth_soft_r2), 4),
            "soft_beta_std": round(soft_beta_std, 4),  # sign carries Wn>0 -> Su<0
        },
    }


def main() -> None:
    stats = fit()
    STATS_PATH.write_text(json.dumps(stats, indent=1))
    c = stats["coupling"]
    print(f"wrote {STATS_PATH}")
    print(f"  cheap PC1 explained var : {stats['cheap_explained_var'][0]:.3f}")
    print(
        f"  PC1 loadings            : {dict(zip(stats['cheap_columns'], stats['cheap_pc1']))}"
    )
    print(
        f"  depth R2 -> +softness    : {c['depth_r2']:.3f} -> {c['depth_soft_r2']:.3f}"
    )
    print(f"  softness beta (per SD)  : {c['soft_beta_std']:+.3f} on log Su")


if __name__ == "__main__":
    main()
