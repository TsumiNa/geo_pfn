"""Whole-borehole holdout eval: ``python -m geo_pfn.haneda.holdout_eval``.

Scores a geo-PFN checkpoint (and optional TabICL / hgbt references) on the real
Haneda data with the deployment-faithful protocol: 5-fold GroupKFold over whole
boreholes (seed 42); for each test borehole geo-PFN predicts from its nearest
*whole* neighbour boreholes (coherent context), TabICL/hgbt refit per fold.
Predictions are pooled across folds and scored by RMSE per (model, feature set).

This is the harness for the prior-realism A/B (old vs new prior) and the capacity
sweep (2M vs bigger checkpoints) — every model is scored identically. Run in the
TabICL overlay env when comparing against TabICL::

    uv run --with tabicl python -m geo_pfn.haneda.holdout_eval \
        --checkpoint checkpoints/geopfn2stage_realistic.pt --feature-sets L,LCSG
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
class HoldoutConfig:
    data_path: str = "data/pilot_Su_domain_block_mod4Liu.csv"
    out_path: str = "results/haneda/holdout_eval.json"
    checkpoint: str = "checkpoints/geopfn2stage_realistic.pt"
    device: str = "auto"
    feature_sets: tuple[str, ...] = ("L", "LCSG")
    models: tuple[str, ...] = ("geopfn", "tabicl", "hgbt")
    n_folds: int = 5
    seed: int = 42
    coherent: dict = field(
        default_factory=lambda: {"n_holes": 8, "n_candidates": 24, "n_ensembles": 8}
    )

    def __post_init__(self) -> None:
        self.feature_sets = tuple(self.feature_sets)
        self.models = tuple(self.models)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run(config: HoldoutConfig) -> dict:
    t0 = time.monotonic()
    df = load_haneda(config.data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    bores = df[GROUP].to_numpy()
    folds = borehole_folds(bores, config.n_folds, config.seed)
    device = torch.device(
        config.device
        if config.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    geopfn = load_geopfn(config.checkpoint, device) if "geopfn" in config.models else None
    coh = CoherentConfig(**config.coherent)

    records: list[dict] = []
    for fs_name in config.feature_sets:
        fs = FeatureSet(fs_name)
        x = df[feature_columns(fs)].to_numpy(dtype=np.float64)
        for model in config.models:
            pred = np.full(len(df), np.nan)
            for tr, te in folds:
                if model == "geopfn":
                    for b in np.unique(bores[te]):
                        q = te[bores[te] == b]
                        pred[q] = predict_geopfn_coherent(
                            geopfn, x[tr], su[tr], bores[tr], x[q], coh, 42 + int(b),
                            device,
                        )
                elif model == "tabicl":
                    from tabicl import TabICLRegressor

                    est = TabICLRegressor(
                        device=None if device.type == "cpu" else device.type,
                        random_state=42,
                    )
                    est.fit(x[tr], su[tr])
                    pred[te] = np.asarray(est.predict(x[te]))
                else:
                    from geo_pfn.haneda.runners import make_baseline

                    est = make_baseline(model, "regression")
                    est.fit(x[tr], su[tr])
                    pred[te] = np.asarray(est.predict(x[te]))
            ok = ~np.isnan(pred)
            rec = {
                "model": model,
                "feature_set": fs_name,
                "rmse": round(_rmse(su[ok], pred[ok]), 3),
                "r2": round(_r2(su[ok], pred[ok]), 4),
                "n": int(ok.sum()),
            }
            records.append(rec)
            print(
                f"[{time.monotonic() - t0:6.1f}s] {model:7s} {fs_name:5s} "
                f"rmse={rec['rmse']:6.2f} r2={rec['r2']:.3f}",
                flush=True,
            )

    payload = {"config": asdict(config), "checkpoint": config.checkpoint, "records": records}
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\n## {config.checkpoint}")
    for fs_name in config.feature_sets:
        row = {r["model"]: r["rmse"] for r in records if r["feature_set"] == fs_name}
        print(f"  {fs_name:5s} " + "  ".join(f"{m}={v}" for m, v in row.items()))
    print(f"done in {time.monotonic() - t0:.1f}s -> {config.out_path}", flush=True)
    return payload


def main() -> None:
    d = HoldoutConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=str, default=d.data_path)
    p.add_argument("--out", type=str, default=d.out_path)
    p.add_argument("--checkpoint", type=str, default=d.checkpoint)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--feature-sets", type=str, default=",".join(d.feature_sets))
    p.add_argument("--models", type=str, default=",".join(d.models))
    a = p.parse_args()
    run(
        HoldoutConfig(
            data_path=a.data_path,
            out_path=a.out,
            checkpoint=a.checkpoint,
            device=a.device,
            feature_sets=tuple(t for f in a.feature_sets.split(",") if (t := f.strip())),
            models=tuple(t for m in a.models.split(",") if (t := m.strip())),
        )
    )


if __name__ == "__main__":
    main()
