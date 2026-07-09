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


@dataclass(kw_only=True)
class CoherentConfig:
    """Coherent-context inference: whole nearest boreholes, not random rows.

    The model was trained on *sites* — a few boreholes, each with many rows,
    sharing a spatial field — so the inference context must have the same shape.
    Each ensemble draw takes ``n_holes`` whole boreholes sampled from the
    ``n_candidates`` nearest (by coordinate) to the target, plus the target's own
    anchor rows. Random single-row subsamples (``EnsembleConfig``) break this
    borehole-unit structure and make the model regress to the mean.
    """

    n_holes: int = 8  # neighbour boreholes per context draw
    n_candidates: int = 24  # sample n_holes from this many nearest boreholes
    n_ensembles: int = 8
    query_chunk: int = 512
    coord_idx: tuple[int, int] = (1, 2)  # X, Y columns (depth is col 0)


def _forward_context(
    model: GeoPFN,
    cx: torch.Tensor,
    cy: torch.Tensor,
    query: torch.Tensor,
    query_chunk: int,
    device: torch.device,
) -> torch.Tensor:
    """One ensemble member: context (cx, cy) + query -> predictions in y units."""
    mean, std = cy.mean(), cy.std(unbiased=False).clamp(min=1e-6)
    z_ctx = (cy - mean) / std
    n_c = len(cx)
    preds = []
    for start in range(0, len(query), query_chunk):
        chunk = query[start : start + query_chunk]
        x = torch.cat([cx, chunk]).unsqueeze(0).to(device)
        z = torch.cat([z_ctx, torch.zeros(len(chunk))]).unsqueeze(0).to(device)
        mask = torch.zeros(1, n_c + len(chunk), dtype=torch.bool, device=device)
        mask[0, :n_c] = True
        with torch.no_grad():
            logits = model(x, z, mask)
            pred_z, _ = model.bar.mean_std(logits[0, n_c:])
        preds.append(pred_z.cpu() * std + mean)
    return torch.cat(preds)


def predict_geopfn_coherent(
    model: GeoPFN,
    x_context: np.ndarray,
    y_context: np.ndarray,
    ctx_bores: np.ndarray,
    x_query: np.ndarray,
    config: CoherentConfig,
    seed: int,
    device: torch.device,
    anchor_x: np.ndarray | None = None,
    anchor_y: np.ndarray | None = None,
) -> np.ndarray:
    """Predict a target borehole's query rows from its nearest whole boreholes.

    ``ctx_bores`` gives the borehole id of each context row; the target's own
    coordinates are read from ``x_query`` (columns ``config.coord_idx``). Anchor
    rows, if given, are added to every draw. Predictions are averaged in the
    original target units.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    cx_i, cy_i = config.coord_idx
    query_xy = x_query[0, [cx_i, cy_i]]

    bores = np.unique(ctx_bores)
    bore_xy = np.array([x_context[ctx_bores == b][0, [cx_i, cy_i]] for b in bores])
    order = np.argsort(((bore_xy - query_xy) ** 2).sum(1))
    nearest = bores[order[: config.n_candidates]]

    ax = None if anchor_x is None else torch.tensor(anchor_x, dtype=torch.float32)
    ay = None if anchor_y is None else torch.tensor(anchor_y, dtype=torch.float32)
    query = torch.tensor(x_query, dtype=torch.float32)
    total = torch.zeros(len(query))

    for _ in range(config.n_ensembles):
        n = min(config.n_holes, len(nearest))
        pick = nearest[torch.randperm(len(nearest), generator=gen)[:n].numpy()]
        rows = np.isin(ctx_bores, pick)
        cx = torch.tensor(x_context[rows], dtype=torch.float32)
        cy = torch.tensor(y_context[rows], dtype=torch.float32)
        if ax is not None and len(ax):
            cx, cy = torch.cat([ax, cx]), torch.cat([ay, cy])
        total += _forward_context(model, cx, cy, query, config.query_chunk, device)

    return (total / config.n_ensembles).numpy()


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
        mean, std = cy.mean(), cy.std(unbiased=False).clamp(min=1e-6)
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
