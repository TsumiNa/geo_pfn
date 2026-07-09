"""Predict a Su grid with TabICL for visualization: ``python -m geo_pfn.haneda.su_grid``.

TabICL (the best non-TabPFN model) fits on all real specimens using location +
depth only (feature set L — a virtual grid point has no measured geotech) and
predicts Su on a regular x-y-depth grid. Run in the overlay env:

    uv run --with tabicl python -m geo_pfn.haneda.su_grid --device mps

Output parquet columns: ``x, y, depth, su_pred`` (depth positive, metres below
surface). Default grid: 100 x 100 over the borehole x-y extent, depths 5..70 m
every 5 m -> 140,000 predictions.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from geo_pfn.haneda.data import TARGET, load_haneda


def build_grid(df: pd.DataFrame, nx: int, ny: int, depths: np.ndarray) -> np.ndarray:
    """Grid feature rows [depth_m, X, Y] over the borehole x-y extent."""
    xs = np.linspace(df["X"].min(), df["X"].max(), nx)
    ys = np.linspace(df["Y"].min(), df["Y"].max(), ny)
    dd, xx, yy = np.meshgrid(-depths, xs, ys, indexing="ij")  # depth_m negative
    return np.column_stack([dd.ravel(), xx.ravel(), yy.ravel()])


def run(
    data_path: str,
    out_path: str,
    device: str,
    nx: int,
    ny: int,
    depth_step: int,
    depth_max: int,
    chunk: int,
) -> None:
    from tabicl import TabICLRegressor  # ephemeral overlay dependency

    df = load_haneda(data_path)
    x_fit = df[["depth_m", "X", "Y"]].to_numpy(dtype=np.float64)  # feature set L
    y_fit = df[TARGET].to_numpy(dtype=np.float64)

    depths = np.arange(depth_step, depth_max + 1, depth_step)
    grid = build_grid(df, nx, ny, depths)
    print(
        f"fit on {len(x_fit)} specimens (L features); predicting {len(grid):,} "
        f"grid points ({nx}x{ny} x-y, depths {depths.min()}-{depths.max()}m)",
        flush=True,
    )

    reg = TabICLRegressor(
        device=None if device == "auto" else device, kv_cache=True, random_state=42
    )
    t0 = time.monotonic()
    reg.fit(x_fit, y_fit)
    preds = []
    for start in range(0, len(grid), chunk):
        preds.append(np.asarray(reg.predict(grid[start : start + chunk])))
        print(
            f"  [{time.monotonic() - t0:6.1f}s] {start + chunk:>7,}/{len(grid):,}",
            flush=True,
        )
    su = np.concatenate(preds)

    out = pd.DataFrame(
        {
            "x": grid[:, 1],
            "y": grid[:, 2],
            "depth": -grid[:, 0],  # positive metres below surface
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
    parser.add_argument(
        "--data-path", type=str, default="data/pilot_Su_domain_block_mod4Liu.csv"
    )
    parser.add_argument(
        "--out", type=str, default="results/haneda/su_grid_tabicl.parquet"
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--depth-step", type=int, default=5)
    parser.add_argument("--depth-max", type=int, default=70)
    parser.add_argument("--chunk", type=int, default=5000)
    args = parser.parse_args()
    run(
        data_path=args.data_path,
        out_path=args.out,
        device=args.device,
        nx=args.nx,
        ny=args.ny,
        depth_step=args.depth_step,
        depth_max=args.depth_max,
        chunk=args.chunk,
    )


if __name__ == "__main__":
    main()
