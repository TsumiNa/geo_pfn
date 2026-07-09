"""Hypothesis test: geo-PFN on the real Haneda anchor task.

Runs the trained two-stage geo-PFN through the same sparse-anchor cross-borehole
transfer task as ``geo_pfn.haneda.anchor`` (same borehole folds, same k, same
depth-stratified anchors), so its paired anchor-holdout gain and absolute
RMSE-with-anchors are comparable to the committed v2 (k=5: -1.92 / 12.46) and
TabICL (-1.43) numbers.

Each test borehole is predicted separately with context = a random subsample of
the training-fold rows (neighbours) plus, in the anchor arm, that borehole's k
anchors (forced into every ensemble draw). This matches both the model's
pretraining regime (sites of a few hundred rows) and the real use case (predict
one virtual borehole from its neighbours).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geo_pfn.geopfn.model import GeoPFN, GeoPFNConfig
from geo_pfn.geopfn.predict import CoherentConfig, predict_geopfn_coherent
from geo_pfn.haneda.anchor import select_anchors
from geo_pfn.haneda.data import (
    GROUP,
    TARGET,
    FeatureSet,
    borehole_folds,
    feature_columns,
    load_haneda,
)
from geo_pfn.haneda.runners import regression_metrics
from geo_pfn.util import resolve_device


def load_geopfn(path: str, device: torch.device) -> GeoPFN:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GeoPFN(GeoPFNConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def run(
    checkpoint: str,
    data_path: str,
    out_path: str,
    device_name: str,
    feature_set: FeatureSet,
    target_col: str,
    k_anchors: tuple[int, ...],
    n_folds: int,
    seed: int,
    coh: CoherentConfig,
) -> None:
    device = resolve_device(device_name)
    model = load_geopfn(checkpoint, device)
    df = load_haneda(data_path)
    target = df[target_col].to_numpy(dtype=np.float64)
    bores = df[GROUP].to_numpy()
    folds = borehole_folds(bores, n_folds, seed)

    feats = [c for c in feature_columns(feature_set) if c != target_col]
    x_full = df[feats].to_numpy(dtype=np.float64)  # native NaN
    t0 = time.monotonic()
    records = []
    for k in k_anchors:
        for fold, (train_idx, test_idx) in enumerate(folds):
            gen = np.random.default_rng(seed + 100 * k + fold)
            anchor_idx, query_idx = select_anchors(df, test_idx, k, gen)
            for arm in ("holdout", "anchor"):
                preds, truths = [], []
                for bore in np.unique(bores[query_idx]):
                    q = query_idx[bores[query_idx] == bore]
                    ax = ay = None
                    if arm == "anchor":
                        a = anchor_idx[bores[anchor_idx] == bore]
                        ax, ay = x_full[a], target[a]
                    pred = predict_geopfn_coherent(
                        model,
                        x_full[train_idx],
                        target[train_idx],
                        bores[train_idx],
                        x_full[q],
                        coh,
                        seed + int(bore),
                        device,
                        anchor_x=ax,
                        anchor_y=ay,
                    )
                    preds.append(pred)
                    truths.append(target[q])
                metrics = regression_metrics(
                    np.concatenate(truths), np.concatenate(preds)
                )
                records.append(
                    {
                        "model": "geopfn",
                        "arm": arm,
                        "feature_set": feature_set.value,
                        "target": target_col,
                        "k_anchors": k,
                        "fold": fold,
                        "n_query": len(query_idx),
                        **metrics,
                    }
                )
                print(
                    f"[{time.monotonic() - t0:7.1f}s] geopfn {arm:7s} k={k} fold={fold} "
                    f"rmse={metrics['rmse']:.3f} r2={metrics['r2']:.3f}",
                    flush=True,
                )
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "config": {
                        "checkpoint": checkpoint,
                        "feature_set": feature_set.value,
                        "target": target_col,
                        "k_anchors": list(k_anchors),
                        "split": f"GroupKFold(borehole, n_splits={n_folds}, seed={seed})",
                        "context": "coherent (nearest whole boreholes)",
                        "ensemble": {
                            "n_holes": coh.n_holes,
                            "n_candidates": coh.n_candidates,
                            "n_ensembles": coh.n_ensembles,
                        },
                    },
                    "records": records,
                }
                Path(out_path).write_text(json.dumps(payload, indent=1))
    _summary(records)
    print(f"done in {time.monotonic() - t0:.1f}s -> {out_path}", flush=True)


def _summary(records: list[dict]) -> None:
    print(
        "\n## geo-PFN anchor - holdout (RMSE, paired per fold; negative = anchors help)"
    )
    for k in sorted({r["k_anchors"] for r in records}):
        by_fold: dict[int, dict[str, float]] = {}
        for r in records:
            if r["k_anchors"] == k:
                by_fold.setdefault(r["fold"], {})[r["arm"]] = r["rmse"]
        diffs = [v["anchor"] - v["holdout"] for v in by_fold.values() if len(v) == 2]
        hold = [v["holdout"] for v in by_fold.values() if "holdout" in v]
        anch = [v["anchor"] for v in by_fold.values() if "anchor" in v]
        d = np.array(diffs)
        sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(
            f"  k={k}: holdout={np.mean(hold):.3f} anchor={np.mean(anch):.3f} "
            f"diff={d.mean():+.3f} +- {sem:.3f} (n={len(d)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/geopfn2stage.pt")
    parser.add_argument(
        "--data-path", type=str, default="data/pilot_Su_domain_block_mod4Liu.csv"
    )
    parser.add_argument("--out", type=str, default="results/haneda/anchor_geopfn.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--feature-set", type=str, default=FeatureSet.LCSG.value)
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--k-anchors", type=str, default="1,2,3,5")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-holes", type=int, default=8)
    parser.add_argument("--n-candidates", type=int, default=24)
    parser.add_argument("--n-ensembles", type=int, default=8)
    args = parser.parse_args()
    run(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        out_path=args.out,
        device_name=args.device,
        feature_set=FeatureSet(args.feature_set),
        target_col=args.target,
        k_anchors=tuple(int(t) for k in args.k_anchors.split(",") if (t := k.strip())),
        n_folds=args.n_folds,
        seed=args.seed,
        coh=CoherentConfig(
            n_holes=args.n_holes,
            n_candidates=args.n_candidates,
            n_ensembles=args.n_ensembles,
        ),
    )


if __name__ == "__main__":
    main()
