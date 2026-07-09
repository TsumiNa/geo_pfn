"""Two-stage TabICL geology-ablation Su grids (visual-only): ``python -m
geo_pfn.haneda.su_grid_ablation``.

Runs in the overlay env::

    uv run --with tabicl python -m geo_pfn.haneda.su_grid_ablation --device cpu

Motivation: the user asked whether large-scale geology (soil + grain size) lifts
local Su prediction, visualised on the same 100x100 x-y x depth grid as
``su_grid``. Two arms:

- ``geology``  — feature set LSG: location + soil code + grain-size fractions;
- ``nogeology`` — feature set LC: location + cheap geotech (Wn, Gs, LL, PL,
  rho_t, e), geology removed.

**CAVEAT (why this is visual-only, not a real ablation).** A virtual grid point
has no measured geotech/geology, so each arm's non-location features must first
be *spatially imputed* from (x, y, depth) before Su is predicted. The imputed
features are then deterministic functions of the coordinates and carry no
information beyond (x, y, depth); the two grids therefore *visualise* each feature
set's Su field but cannot demonstrate that geology helps. A genuine feature
ablation must be run where geology is independently measured — the real specimens
(whole-borehole holdout), not a dense grid.

Stage 1 (impute features on the grid) uses a fast gradient-boosting spatial
regressor/classifier — TabICL forward on 140k rows is far too slow to call once
per feature. Stage 2 (predict Su) uses TabICL, matching ``su_grid``: TabICL is the
model under comparison, the stage-1 fills are just plausible feature values.

Output parquet columns: ``x, y, depth, su_pred`` (depth positive, metres). Not
committed (results/haneda/*.parquet is gitignored).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from geo_pfn.haneda.data import (
    CHEAP_COLUMNS,
    GRAIN_COLUMNS,
    SOIL_CODE,
    TARGET,
    load_haneda,
)

# feature layout per arm: (location cols, geology/cheap cols, which are categorical)
LOC = ["depth_m", "X", "Y"]
ARMS = {
    "geology": [SOIL_CODE, *GRAIN_COLUMNS],  # LSG minus location
    "nogeology": list(CHEAP_COLUMNS),  # LC minus location
}


def _build_grid(df: pd.DataFrame, nx: int, ny: int, depths: np.ndarray) -> np.ndarray:
    xs = np.linspace(df["X"].min(), df["X"].max(), nx)
    ys = np.linspace(df["Y"].min(), df["Y"].max(), ny)
    dd, xx, yy = np.meshgrid(-depths, xs, ys, indexing="ij")  # depth_m negative
    return np.column_stack([dd.ravel(), xx.ravel(), yy.ravel()])  # [depth_m, X, Y]


def _predict_chunked(reg, x: np.ndarray, chunk: int) -> np.ndarray:
    out = [np.asarray(reg.predict(x[s : s + chunk])) for s in range(0, len(x), chunk)]
    return np.concatenate(out)


def _impute_feature(
    col: str,
    df: pd.DataFrame,
    loc_fit: np.ndarray,
    grid_loc: np.ndarray,
) -> np.ndarray:
    """Stage 1: spatially predict one feature on the grid from (x, y, depth).

    Uses a fast gradient-boosting model (not TabICL) — this is only filling in
    plausible feature values, and TabICL on 140k rows is far too slow to call per
    feature. Soil code is categorical, everything else is regression.
    """
    from geo_pfn.haneda.runners import make_baseline

    observed = df[col].notna().to_numpy()
    y = df.loc[observed, col].to_numpy()
    task = "classification" if col == SOIL_CODE else "regression"
    est = make_baseline("hgbt", task)
    est.fit(loc_fit[observed], y)
    return np.asarray(est.predict(grid_loc))


def run(
    arm: str,
    data_path: str,
    out_path: str,
    device: str,
    nx: int,
    ny: int,
    depth_step: int,
    depth_max: int,
    chunk: int,
) -> None:
    from tabicl import TabICLRegressor

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    extra_cols = ARMS[arm]

    df = load_haneda(data_path)
    loc_fit = df[LOC].to_numpy(dtype=np.float64)
    y_su = df[TARGET].to_numpy(dtype=np.float64)

    depths = np.arange(depth_step, depth_max + 1, depth_step)
    grid_loc = _build_grid(df, nx, ny, depths)
    print(
        f"arm={arm}: features = location + {extra_cols}\n"
        f"fit on {len(df)} specimens; grid = {len(grid_loc):,} points "
        f"({nx}x{ny} x-y, depths {depths.min()}-{depths.max()}m)",
        flush=True,
    )

    t0 = time.monotonic()
    # stage 1: spatially impute each non-location feature on the grid
    grid_extra = []
    for col in extra_cols:
        pred = _impute_feature(col, df, loc_fit, grid_loc)
        grid_extra.append(pred)
        print(f"  [{time.monotonic() - t0:6.1f}s] stage1 imputed {col}", flush=True)
    grid_x = np.column_stack([grid_loc, *grid_extra])

    # stage 2: Su ~ f(location + extra), fit on real specimens (mean-imputed NaN),
    # predict on the grid using the stage-1 imputations
    fit_extra = df[extra_cols].to_numpy(dtype=np.float64)
    col_means = np.nanmean(fit_extra, axis=0)
    fit_extra = np.where(np.isnan(fit_extra), col_means, fit_extra)
    fit_x = np.column_stack([loc_fit, fit_extra])

    reg = TabICLRegressor(
        device=None if device == "auto" else device, kv_cache=True, random_state=42
    )
    reg.fit(fit_x, y_su)
    su = _predict_chunked(reg, grid_x, chunk)
    print(f"  [{time.monotonic() - t0:6.1f}s] stage2 predicted Su", flush=True)

    out = pd.DataFrame(
        {
            "x": grid_loc[:, 1],
            "y": grid_loc[:, 2],
            "depth": -grid_loc[:, 0],
            "su_pred": su,
        }
    )
    out.to_parquet(out_path, index=False)
    print(
        f"saved {len(out):,} rows -> {out_path}  "
        f"(su_pred: {su.min():.1f}..{su.max():.1f}, mean {su.mean():.1f})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=str, default="geology", choices=sorted(ARMS))
    parser.add_argument(
        "--data-path", type=str, default="data/pilot_Su_domain_block_mod4Liu.csv"
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--depth-step", type=int, default=5)
    parser.add_argument("--depth-max", type=int, default=70)
    parser.add_argument("--chunk", type=int, default=5000)
    args = parser.parse_args()
    out_path = args.out or f"results/haneda/su_grid_tabicl_{args.arm}.parquet"
    run(
        arm=args.arm,
        data_path=args.data_path,
        out_path=out_path,
        device=args.device,
        nx=args.nx,
        ny=args.ny,
        depth_step=args.depth_step,
        depth_max=args.depth_max,
        chunk=args.chunk,
    )


if __name__ == "__main__":
    main()
