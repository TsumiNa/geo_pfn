"""Pack sparse-grid predictions into the self-contained 3D demo HTML.

Reads the outputs of :mod:`geo_pfn.haneda.sparse_grid` plus the pilot CSV and
emits a single-file business demo (``sparse_demo_template.html`` with the data
and a local copy of three.js injected). Model names are deliberately mapped to
neutral labels (``ours`` = geo-PFN, ``other`` = TabICL) — the demo must not
reveal engine identities.

    uv run python -m geo_pfn.haneda.sparse_demo \
        --three-js /path/to/three.min.js --out results/haneda/su_demo.html

Predictions are quantized to uint8 over a shared robust range (0.5th-99.5th
percentile of all predictions and measured Su) and embedded as base64 — about
7.5 MB for 20 groups x 2 models, fine for a local file.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geo_pfn.haneda.data import GROUP, SOIL_COLUMN, TARGET, load_haneda

MODEL_LABELS = {"ours": "geopfn", "other": "tabicl"}  # never shown in the HTML


def quantize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """uint8 quantization of ``values`` over [lo, hi] (clipped)."""
    q = np.round((np.clip(values, lo, hi) - lo) / (hi - lo) * 255.0)
    return q.astype(np.uint8)


def build_data(grid_dir: Path, data_path: str) -> dict:
    """Assemble the DATA object embedded in the demo HTML.

    Requires ``ground.json`` in ``grid_dir`` — per-grid-node and per-specimen
    ground elevation (m, AP), harvested from the team's elevation-aware viewer
    and resampled onto our grid (see the 2026-07-15 session notes).
    """
    meta = json.loads((grid_dir / "meta.json").read_text())
    ground_path = grid_dir / "ground.json"
    if not ground_path.exists():
        raise FileNotFoundError(
            f"{ground_path} missing — harvest ground elevation first "
            "(grid ge + per-specimen ge)"
        )
    ground = json.loads(ground_path.read_text())
    df = load_haneda(data_path)
    su = df[TARGET].to_numpy(dtype=np.float64)
    depth = -df["depth_m"].to_numpy(dtype=np.float64)  # positive metres down
    if depth.min() < 0:
        raise ValueError("expected depth_m <= 0 (below surface) in the pilot CSV")

    preds: dict[str, list[np.ndarray]] = {label: [] for label in MODEL_LABELS}
    for g in range(len(meta["groups"])):
        for label, model in MODEL_LABELS.items():
            preds[label].append(np.load(grid_dir / f"pred_g{g:02d}_{model}.npy"))

    stacked = np.concatenate([np.concatenate(v) for v in preds.values()] + [su])
    lo = float(np.percentile(stacked, 0.5))
    hi = float(np.percentile(stacked, 99.5))

    x0, x1, nx = meta["grid"]["xs"]
    y0, y1, ny = meta["grid"]["ys"]
    groups = [
        {
            "boreholes": gm["boreholes"],
            "frac": gm["frac_actual"],
            "rmse": {
                label: gm["models"][model]["rmse_heldout"]
                for label, model in MODEL_LABELS.items()
            },
            "real_secs": {
                label: gm["models"][model]["seconds"]
                for label, model in MODEL_LABELS.items()
            },
        }
        for gm in meta["groups"]
    ]
    return {
        "grid": {
            "x0": x0, "x1": x1, "nx": int(nx),
            "y0": y0, "y1": y1, "ny": int(ny),
            "depths": meta["grid"]["depths"],
        },
        "su_range": [round(lo, 2), round(hi, 2)],
        "specimens": {
            "x": [round(float(v), 1) for v in df["X"]],
            "y": [round(float(v), 1) for v in df["Y"]],
            "depth": [round(float(v), 2) for v in depth],
            "su": [round(float(v), 1) for v in su],
            "bore": [int(b) for b in df[GROUP]],
            "soil": ["—" if pd.isna(s) else str(s) for s in df[SOIL_COLUMN]],
            "wn": [None if pd.isna(v) else round(float(v), 1) for v in df["Wn"]],
            "e": [None if pd.isna(v) else round(float(v), 2) for v in df["e"]],
            "ge": ground["spec_ge"],
        },
        "ground": {"ge": ground["ge"]},
        "groups": groups,
        "pred": {
            label: [
                base64.b64encode(quantize(p, lo, hi).tobytes()).decode("ascii")
                for p in group_preds
            ]
            for label, group_preds in preds.items()
        },
    }


def render_html(data: dict, template_path: Path, three_js_path: Path) -> str:
    template = template_path.read_text()
    three_js = three_js_path.read_text()
    if "</script>" in three_js:
        raise ValueError("three.js bundle contains '</script>' — cannot inline")
    payload = json.dumps(data, separators=(",", ":"))
    html = template.replace("/*__THREE__*/", three_js, 1)
    html = html.replace("/*__DATA__*/", payload, 1)
    if "/*__THREE__*/" in html or "/*__DATA__*/" in html:
        raise ValueError("template markers not fully replaced")
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir", type=str, default="results/haneda/sparse_grid"
    )
    parser.add_argument(
        "--data-path", type=str, default="data/pilot_Su_domain_block_mod4Liu.csv"
    )
    parser.add_argument("--three-js", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/haneda/su_demo.html")
    args = parser.parse_args()

    data = build_data(Path(args.grid_dir), args.data_path)
    template = Path(__file__).with_name("sparse_demo_template.html")
    html = render_html(data, template, Path(args.three_js))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({len(html)/1e6:.1f} MB, {len(data['groups'])} groups)")


if __name__ == "__main__":
    main()
