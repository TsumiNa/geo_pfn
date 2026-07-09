"""Soil-type classification under the anchor protocol: ``python -m geo_pfn.haneda.anchor_soil``.

Companion to ``anchor.py`` (regression). Asks how well the *lithology*
(soil_B02) at unmeasured depths can be predicted from location + cheap geotech
(+ neighbours), with and without a few depth-spread anchors from the target
borehole. The geo-PFN is regression-only, so this uses the classification-
capable models (TabICL, HistGradientBoosting); a geo-PFN soil head is
documented future work (docs/geo-scm-design.md §9.1).

Target is the grouped soil code (n>=30 classes + "other", from data.encode_soil);
features are location + cheap geotech + grain size (soil excluded).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from geo_pfn.haneda.anchor import select_anchors
from geo_pfn.haneda.data import (
    CHEAP_COLUMNS,
    GRAIN_COLUMNS,
    GROUP,
    LOCATION_COLUMNS,
    borehole_folds,
    encode_soil,
    load_haneda,
)
from geo_pfn.haneda.runners import classification_metrics, make_baseline
from geo_pfn.util import resolve_device

FEATURES = list(LOCATION_COLUMNS) + list(CHEAP_COLUMNS) + list(GRAIN_COLUMNS)


def _predict(model: str, x_fit, y_fit, x_query, device):
    if model == "tabicl":
        from tabicl import TabICLClassifier

        est = TabICLClassifier(device=None if device == "auto" else device)
    else:
        est = make_baseline(model, "classification")
    est.fit(x_fit, y_fit)
    return np.asarray(est.predict(x_query))


def run(
    data_path: str,
    out_path: str,
    device_name: str,
    models: tuple[str, ...],
    k_anchors: tuple[int, ...],
    n_folds: int,
    seed: int,
) -> None:
    device = str(resolve_device(device_name))
    df = load_haneda(data_path)
    x_full = df[FEATURES].to_numpy(dtype=np.float64)
    soil = encode_soil(df["soil_B02"]).astype(int)  # grouped soil code
    folds = borehole_folds(df[GROUP].to_numpy(), n_folds, seed)

    t0 = time.monotonic()
    records = []
    for k in k_anchors:
        for fold, (train_idx, test_idx) in enumerate(folds):
            gen = np.random.default_rng(seed + 100 * k + fold)
            anchor_idx, query_idx = select_anchors(df, test_idx, k, gen)
            arms = (
                ("holdout", train_idx),
                ("anchor", np.concatenate([train_idx, anchor_idx])),
            )
            for arm, fit_idx in arms:
                for model in models:
                    pred = _predict(
                        model, x_full[fit_idx], soil[fit_idx], x_full[query_idx], device
                    )
                    metrics = classification_metrics(soil[query_idx], pred)
                    records.append(
                        {
                            "experiment": "anchor_soil",
                            "model": model,
                            "arm": arm,
                            "k_anchors": k,
                            "fold": fold,
                            "n_query": len(query_idx),
                            **metrics,
                        }
                    )
                    print(
                        f"[{time.monotonic() - t0:7.1f}s] {model:8s} {arm:7s} k={k} "
                        f"fold={fold} acc={metrics['accuracy']:.3f} "
                        f"f1={metrics['macro_f1']:.3f}",
                        flush=True,
                    )
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_text(json.dumps(records, indent=1))
    _summary(records)
    print(f"done in {time.monotonic() - t0:.1f}s -> {out_path}", flush=True)


def _summary(records: list[dict]) -> None:
    print("\n## soil classification: accuracy by model, arm, k")
    for model in dict.fromkeys(r["model"] for r in records):
        for k in sorted({r["k_anchors"] for r in records}):
            for arm in ("holdout", "anchor"):
                accs = [
                    r["accuracy"]
                    for r in records
                    if r["model"] == model and r["k_anchors"] == k and r["arm"] == arm
                ]
                if accs:
                    print(
                        f"  {model:8s} k={k} {arm:7s} acc={np.mean(accs):.3f} "
                        f"± {np.std(accs, ddof=1) / np.sqrt(len(accs)):.3f}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=str, default="data/pilot_Su_domain_block_mod4Liu.csv"
    )
    parser.add_argument("--out", type=str, default="results/haneda/anchor_soil.json")
    parser.add_argument("--device", type=str, default="auto")
    # add "tabicl" via --models (needs `uv run --with tabicl`)
    parser.add_argument("--models", type=str, default="hgbt")
    parser.add_argument("--k-anchors", type=str, default="0,3")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # k=0 means holdout only (no anchors); select_anchors needs k>=1, so map 0 -> skip
    ks = tuple(
        int(t) for k in args.k_anchors.split(",") if (t := k.strip()) and int(t) > 0
    )
    run(
        data_path=args.data_path,
        out_path=args.out,
        device_name=args.device,
        models=tuple(t for m in args.models.split(",") if (t := m.strip())),
        k_anchors=ks or (3,),
        n_folds=args.n_folds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
