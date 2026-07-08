"""Inference adapter for the two-stage geo-PFN.

The model does fit+predict in one forward pass over a single (context + query)
table, so there is no training at predict time — the context rows are the
"fit" data. It was pretrained on sites of <= a few hundred rows, so a large
context (e.g. a whole Haneda training fold) is consumed by ensembling: each
forward uses a random subset of context rows (always including the caller's
required-context rows, e.g. a target borehole's anchors), and the per-query
bar-distribution predictions are averaged in normalized space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from geo_pfn.geopfn.model import GeoPFN


@dataclass(kw_only=True)
class EnsembleConfig:
    ctx_size: int = 256  # context rows per forward
    n_ensembles: int = 8
    query_chunk: int = 512


def predict_geopfn(
    model: GeoPFN,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_query: np.ndarray,
    config: EnsembleConfig,
    seed: int,
    device: torch.device,
    keep_context: np.ndarray | None = None,
) -> np.ndarray:
    """Predict query targets by ensembling context subsamples.

    ``keep_context`` (indices into the context arrays) are forced into every
    subsample — use it for anchor rows that must always be present. Returns the
    query predictions in the original target units.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    n_ctx = len(x_context)
    keep = np.array([], dtype=int) if keep_context is None else np.asarray(keep_context)
    pool = np.setdiff1d(np.arange(n_ctx), keep)

    ctx_x_all = torch.tensor(x_context, dtype=torch.float32)
    ctx_y_all = torch.tensor(y_context, dtype=torch.float32)
    query = torch.tensor(x_query, dtype=torch.float32)
    total = torch.zeros(len(query))

    for e in range(config.n_ensembles):
        n_extra = max(0, min(config.ctx_size - len(keep), len(pool)))
        perm = torch.randperm(len(pool), generator=gen)[:n_extra].numpy()
        idx = np.concatenate([keep, pool[perm]]).astype(int)
        cx, cy = ctx_x_all[idx], ctx_y_all[idx]
        mean, std = cy.mean(), cy.std().clamp(min=1e-6)
        z_ctx = (cy - mean) / std

        preds = []
        for start in range(0, len(query), config.query_chunk):
            chunk = query[start : start + config.query_chunk]
            x = torch.cat([cx, chunk]).unsqueeze(0).to(device)
            n_c = len(cx)
            z = torch.cat([z_ctx, torch.zeros(len(chunk))]).unsqueeze(0).to(device)
            ctx_mask = torch.zeros(
                1, len(cx) + len(chunk), dtype=torch.bool, device=device
            )
            ctx_mask[0, :n_c] = True
            with torch.no_grad():
                logits = model(x, z, ctx_mask)
                pred_z, _ = model.bar.mean_std(logits[0, n_c:])
            preds.append(pred_z.cpu() * std + mean)
        total += torch.cat(preds)

    return (total / config.n_ensembles).numpy()
