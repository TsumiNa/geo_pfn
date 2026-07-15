"""Sparse-context Su grids for the 3D demo: ``python -m geo_pfn.haneda.sparse_grid``.

The deployment story behind the demo: a client has drilled only a *sparse*
survey (5-10% of the Haneda boreholes) and wants the full 3D Su field. For each
of ``n_groups`` random sparse surveys, TabICL and geo-PFN both predict Su on the
same fine grid as :mod:`su_grid` (100x100 x-y, depths 5..70 m every 5 m ->
140,000 points) plus at every real specimen location (so each group carries an
honest held-out RMSE for the narrative).

Context sampling: whole boreholes (the borehole-unit protocol of
:mod:`sparse_context`), accumulated until a target row fraction drawn uniformly
from [0.05, 0.10] is reached.

Run in the overlay env (TabICL)::

    uv run --with tabicl python -m geo_pfn.haneda.sparse_grid --device mps

Outputs under ``results/haneda/sparse_grid/`` (gitignored like all parquet/npy):

- ``meta.json`` — grid axes, group definitions (boreholes, rows, fraction),
  per-model held-out RMSE and wall-clock seconds per group;
- ``pred_g{g:02d}_{model}.npy`` — float32 grid predictions (140,000,);
- ``spec_g{g:02d}_{model}.npy`` — float32 predictions at the held-out specimen
  rows of that group.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geo_pfn.geopfn.eval_anchor import load_geopfn
from geo_pfn.geopfn.predict import CoherentConfig, predict_geopfn_coherent
from geo_pfn.haneda.data import GROUP, TARGET, load_haneda
from geo_pfn.haneda.su_grid import build_grid
from geo_pfn.util import resolve_device


@dataclass(kw_only=True)
class SparseGridConfig:
    data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv"
    out_dir: str = "results/haneda/sparse_grid"
    checkpoint: str = "checkpoints/gen_2m_24k.pt"
    device: str = "mps"
    models: tuple[str, ...] = ("tabicl", "geopfn")
    n_groups: int = 20
    frac_lo: float = 0.05
    frac_hi: float = 0.10
    nx: int = 100
    ny: int = 100
    depth_step: int = 5
    depth_max: int = 70
    tabicl_chunk: int = 10_000
    seed: int = 42
    n_holes: int = 8
    n_ensembles: int = 8

    def __post_init__(self) -> None:
        self.models = tuple(self.models)


def sample_context_boreholes(
    bores: np.ndarray, frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Whole boreholes, shuffled, accumulated until ``frac`` of all rows."""
    unique = np.array(sorted(set(bores)))
    order = rng.permutation(unique)
    counts = pd.Series(bores).value_counts()
    target = frac * len(bores)
    picked: list = []
    total = 0
    for b in order:
        picked.append(b)
        total += int(counts[b])
        if total >= target:
            break
    return np.array(picked)


def run(config: SparseGridConfig) -> None:
    t0 = time.monotonic()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    df = load_haneda(config.data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    bores = df[GROUP].to_numpy()
    x_all = df[["depth_m", "X", "Y"]].to_numpy(dtype=np.float64)  # feature set L

    depths = np.arange(config.depth_step, config.depth_max + 1, config.depth_step)
    grid = build_grid(df, config.nx, config.ny, depths)
    print(f"grid: {len(grid):,} points; specimens: {len(df):,}", flush=True)

    geopfn = (
        load_geopfn(config.checkpoint, device) if "geopfn" in config.models else None
    )

    groups_meta: list[dict] = []
    rng = np.random.default_rng(config.seed)
    for g in range(config.n_groups):
        frac = rng.uniform(config.frac_lo, config.frac_hi)
        ctx_bores_ids = sample_context_boreholes(bores, frac, rng)
        ctx = np.isin(bores, ctx_bores_ids)
        held = ~ctx
        meta: dict = {
            "group": g,
            "frac_target": round(float(frac), 4),
            "frac_actual": round(float(ctx.mean()), 4),
            "boreholes": [int(b) for b in ctx_bores_ids],
            "n_ctx_rows": int(ctx.sum()),
            "models": {},
        }
        print(
            f"[{time.monotonic() - t0:7.1f}s] group {g:02d}: "
            f"{len(ctx_bores_ids)} boreholes, {ctx.sum()} rows "
            f"({ctx.mean():.1%})",
            flush=True,
        )

        for model in config.models:
            t_m = time.monotonic()
            if model == "geopfn":
                coh = CoherentConfig(
                    n_holes=config.n_holes,
                    n_candidates=max(24, len(ctx_bores_ids)),
                    n_ensembles=config.n_ensembles,
                )
                pred_grid = predict_geopfn_coherent(
                    geopfn, x_all[ctx], su[ctx], bores[ctx], grid, coh,
                    config.seed + g, device,
                )
                pred_spec = predict_geopfn_coherent(
                    geopfn, x_all[ctx], su[ctx], bores[ctx], x_all[held], coh,
                    config.seed + g, device,
                )
            elif model == "tabicl":
                from tabicl import TabICLRegressor

                est = TabICLRegressor(
                    device=device.type, kv_cache=True, random_state=42
                )
                est.fit(x_all[ctx], su[ctx])
                parts = [
                    np.asarray(est.predict(grid[s : s + config.tabicl_chunk]))
                    for s in range(0, len(grid), config.tabicl_chunk)
                ]
                pred_grid = np.concatenate(parts)
                pred_spec = np.asarray(est.predict(x_all[held]))
            else:
                raise ValueError(f"unknown model {model}")

            secs = time.monotonic() - t_m
            rmse = float(np.sqrt(np.mean((pred_spec - su[held]) ** 2)))
            np.save(out_dir / f"pred_g{g:02d}_{model}.npy",
                    pred_grid.astype(np.float32))
            np.save(out_dir / f"spec_g{g:02d}_{model}.npy",
                    pred_spec.astype(np.float32))
            meta["models"][model] = {
                "rmse_heldout": round(rmse, 3),
                "seconds": round(secs, 2),
            }
            print(
                f"[{time.monotonic() - t0:7.1f}s]   {model:7s} "
                f"rmse={rmse:6.2f}  ({secs:.1f}s)",
                flush=True,
            )
        groups_meta.append(meta)
        _write_meta(config, out_dir, df, depths, groups_meta)

    print(f"done in {time.monotonic() - t0:.1f}s -> {out_dir}", flush=True)


def _write_meta(
    config: SparseGridConfig,
    out_dir: Path,
    df: pd.DataFrame,
    depths: np.ndarray,
    groups_meta: list[dict],
) -> None:
    meta = {
        "config": asdict(config),
        "grid": {
            "xs": [float(df["X"].min()), float(df["X"].max()), config.nx],
            "ys": [float(df["Y"].min()), float(df["Y"].max()), config.ny],
            "depths": [int(d) for d in depths],
            "order": "depth-major, then x, then y (meshgrid ij)",
        },
        "groups": groups_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    d = SparseGridConfig()
    parser.add_argument("--data-path", type=str, default=d.data_path)
    parser.add_argument("--out-dir", type=str, default=d.out_dir)
    parser.add_argument("--checkpoint", type=str, default=d.checkpoint)
    parser.add_argument("--device", type=str, default=d.device)
    parser.add_argument("--models", type=str, default=",".join(d.models))
    parser.add_argument("--n-groups", type=int, default=d.n_groups)
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()
    run(
        SparseGridConfig(
            data_path=args.data_path,
            out_dir=args.out_dir,
            checkpoint=args.checkpoint,
            device=args.device,
            models=tuple(t for m in args.models.split(",") if (t := m.strip())),
            n_groups=args.n_groups,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
