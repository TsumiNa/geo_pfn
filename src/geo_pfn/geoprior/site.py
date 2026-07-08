"""Multi-borehole "site" sampler (docs/geo-scm-design.md §8).

A training table is a *site*: several boreholes from one locality that share a
site-level SCM (the region's causal law, so holes are comparable) and a smooth
spatial field over coordinates (so nearby holes resemble each other). Tables are
mixed: a fraction ``p_single`` are a single borehole (teaching within-hole depth
/ layer / cheap->target inference); the rest are multi-hole (teaching
cross-borehole transfer). Context follows the anchor scenario: in a multi-hole
site some holes are dense neighbours (fully observed) and some are sparse
targets (a few depth-spread anchors observed, the rest to predict).

Each batch element is one site with its holes concatenated along the row axis;
one ``_scm_map`` call per element gives a shared per-site causal law whose
per-site normalization preserves the cross-hole level differences the spatial
field creates — exactly the signal a geo-tuned row metric must learn.
"""

from __future__ import annotations

import math

import torch

from geo_pfn.geoprior.config import GeoPriorConfig
from geo_pfn.geoprior.prior import (
    GeoBatch,
    _depth_axis,
    _latent_field,
    _randint,
    _scm_map,
    _stratigraphy,
)


def _allocate_rows(cfg: GeoPriorConfig, r: int, g: torch.Generator) -> list[int]:
    """Choose hole count and split ``r`` rows among holes (each >= min_hole_rows)."""
    single = torch.rand(1, generator=g).item() < cfg.p_single
    max_holes = 1 if single else max(2, min(cfg.max_holes, r // cfg.min_hole_rows))
    n_holes = 1 if single else _randint(g, 2, max_holes)
    if n_holes == 1:
        return [r]
    # random proportions, then round to integers summing to r with each >= min
    props = torch.rand(n_holes, generator=g) + 0.3
    counts = (props / props.sum() * r).floor().long().clamp(min=cfg.min_hole_rows)
    while int(counts.sum()) > r:  # trim overflow from the largest holes
        counts[int(counts.argmax())] -= 1
    counts[int(counts.argmax())] += r - int(counts.sum())
    return counts.tolist()


def _anchor_rows(depth_h: torch.Tensor, k: int, g: torch.Generator) -> torch.Tensor:
    """Indices of ``k`` depth-stratified anchor rows within one hole (sorted rows)."""
    r_h = len(depth_h)
    order = torch.argsort(depth_h)
    bins = torch.tensor_split(order, min(k, r_h))
    return torch.stack(
        [b[torch.randint(len(b), (1,), generator=g)].squeeze(0) for b in bins]
    )


def _build_site(
    cfg: GeoPriorConfig, r: int, g: torch.Generator
) -> dict[str, torch.Tensor]:
    """Build one site's rows: scm_input, depth, coords, soil, hole_id, context_mask."""
    counts = _allocate_rows(cfg, r, g)
    n_holes = len(counts)
    # smooth spatial field: a linear map of coordinates shared by the site
    field_map = torch.randn(2, cfg.latent_dim, generator=g) * cfg.site_field_scale
    coords_h = torch.randn(n_holes, 2, generator=g)
    field = coords_h @ field_map  # (n_holes, latent_dim); nearby coords -> similar

    # which holes are sparse targets (only in multi-hole sites)
    n_target = 0 if n_holes == 1 else _randint(g, 1, n_holes - 1)
    target_holes = set(torch.randperm(n_holes, generator=g)[:n_target].tolist())

    s_in, depth_all, coords_all, soil_all, hole_all, ctx_all = [], [], [], [], [], []
    for h, r_h in enumerate(counts):
        depth = _depth_axis(cfg, 1, r_h, g)  # (1, r_h)
        depth_norm = (depth - depth.mean(1, keepdim=True)) / (
            depth.std(1, keepdim=True) + 1e-6
        )
        layer, soil = _stratigraphy(cfg, depth, g)
        s = _latent_field(cfg, depth, depth_norm, layer, g)  # (1, r_h, d)
        s = s + field[h].view(1, 1, -1)  # spatial-field level shift for this hole
        coords = coords_h[h].view(1, 1, 2).expand(1, r_h, 2)
        s_in.append(torch.cat([s, depth_norm.unsqueeze(-1), coords], dim=-1)[0])
        depth_all.append(depth[0])
        coords_all.append(coords[0])
        soil_all.append(soil[0])
        hole_all.append(torch.full((r_h,), h))

        ctx = torch.ones(r_h, dtype=torch.bool)
        if n_holes == 1:  # single hole: random-fraction context split
            frac = cfg.min_train_frac + (cfg.max_train_frac - cfg.min_train_frac) * (
                torch.rand(1, generator=g).item()
            )
            perm = torch.randperm(r_h, generator=g)
            ctx[perm[max(1, int(r_h * frac)) :]] = False
        elif h in target_holes:  # sparse target: keep only depth-spread anchors
            k = _randint(g, cfg.min_anchors, min(cfg.max_anchors, r_h - 1))
            ctx[:] = False
            ctx[_anchor_rows(depth_all[-1], k, g)] = True
        ctx_all.append(ctx)

    return {
        "scm_input": torch.cat(s_in, dim=0),
        "depth": torch.cat(depth_all),
        "coords": torch.cat(coords_all),
        "soil": torch.cat(soil_all),
        "hole_id": torch.cat(hole_all),
        "context_mask": torch.cat(ctx_all),
    }


def sample_geo_site_batch(
    cfg: GeoPriorConfig, batch_size: int, generator: torch.Generator
) -> GeoBatch:
    """Sample a batch of sites (mixed single/multi-borehole) sharing one shape."""
    g = generator
    r = _randint(g, cfg.min_rows, cfg.max_rows)
    n_feat = _randint(g, cfg.min_features, cfg.max_features)

    sites = [_build_site(cfg, r, g) for _ in range(batch_size)]
    scm_input = torch.stack([s["scm_input"] for s in sites])  # (B, R, in)
    depth = torch.stack([s["depth"] for s in sites])
    coords = torch.stack([s["coords"] for s in sites])
    soil = torch.stack([s["soil"] for s in sites])
    hole_id = torch.stack([s["hole_id"] for s in sites])
    context_mask = torch.stack([s["context_mask"] for s in sites])
    b = batch_size

    n_scm_feat = max(1, n_feat - 4)
    feats, target = _scm_map(cfg, scm_input, n_scm_feat, g)

    scale = torch.exp(torch.randn(b, 1, n_scm_feat, generator=g) * 0.6)
    shift = torch.randn(b, 1, n_scm_feat, generator=g)
    feats = feats * scale + shift
    feats = feats + cfg.obs_noise * scale * torch.randn(b, r, n_scm_feat, generator=g)

    x = torch.cat(
        [depth.unsqueeze(-1), coords, soil.float().unsqueeze(-1), feats], dim=-1
    )
    if cfg.block_missing_prob > 0 and n_scm_feat > 1:
        drop = torch.rand(b, 1, n_scm_feat, generator=g) < cfg.block_missing_prob
        keep_one = drop.all(-1, keepdim=True) & (torch.arange(n_scm_feat) == 0).view(
            1, 1, -1
        )
        drop = drop & ~keep_one
        feat_block = x[:, :, 4:].masked_fill(drop.expand(b, r, n_scm_feat), math.nan)
        x = torch.cat([x[:, :, :4], feat_block], dim=-1)

    return GeoBatch(
        x=x,
        y=target,
        depth=depth,
        layer_id=torch.zeros_like(hole_id),  # per-site layer ids are hole-local; unused
        soil_code=soil.long(),
        train_size=int(context_mask[0].sum()),
        hole_id=hole_id,
        context_mask=context_mask,
    )
