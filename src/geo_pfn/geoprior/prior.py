"""geo-SCM prior sampler: one borehole table per batch element.

Generative process (docs/geo-scm-design.md §2): a depth axis is segmented into
soil layers; a latent field evolves gently within a layer and jumps across a
boundary; a random MLP-SCM maps the latent to observed cheap features and a
downstream target. Column 0 is always depth; columns 1-2 are the borehole's
constant coordinates; column 3 is its piecewise-constant soil code; the rest
are cheap-geotech-like features. The target is a deep SCM node (downstream of
the features), mirroring both the SCM "target = deep node" convention and the
physics (strength is a consequence of water content / plasticity / density).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from geo_pfn.geoprior.config import GeoPriorConfig

_ACTIVATIONS = (torch.tanh, torch.relu, torch.sin)


@dataclass
class GeoBatch:
    """A batch of synthetic boreholes sharing one (rows, features) shape."""

    x: (
        torch.Tensor
    )  # (B, R, F); col0 depth, col1-2 coords, col3 soil code; NaN = missing
    y: torch.Tensor  # (B, R) continuous target (Su-like)
    depth: torch.Tensor  # (B, R) depth below surface, metres
    layer_id: torch.Tensor  # (B, R) int layer index within the borehole
    soil_code: torch.Tensor  # (B, R) int soil type per row
    train_size: int  # single-hole tables: rows < train_size are context
    hole_id: torch.Tensor | None = None  # (B, R) borehole index within the site
    context_mask: torch.Tensor | None = None  # (B, R) bool; True = observed y (context)


def _randint(g: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(low, high + 1, (1,), generator=g).item())


def _depth_axis(
    cfg: GeoPriorConfig, b: int, r: int, g: torch.Generator
) -> torch.Tensor:
    """(B, R) increasing depths: Exp-spaced specimens from the surface down."""
    gaps = -torch.log(torch.rand(b, r, generator=g).clamp(min=1e-6)) * cfg.spacing_mean
    gaps[:, 0] = 0.0
    return cfg.surface_depth + torch.cumsum(gaps, dim=1)


def _stratigraphy(
    cfg: GeoPriorConfig, depth: torch.Tensor, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment each borehole into layers; return (layer_id, soil_code), both (B, R)."""
    b, r = depth.shape
    lam = torch.full((b,), cfg.layer_lambda)
    n_layers = (1 + torch.poisson(lam, generator=g).long()).clamp(
        cfg.min_layers, min(cfg.max_layers, r)
    )
    # boundary depths: fractions of each borehole's depth span
    z0, z1 = depth[:, :1], depth[:, -1:]
    cuts = torch.rand(b, cfg.max_layers - 1, generator=g).sort(dim=1).values
    layer_id = torch.zeros(b, r, dtype=torch.long)
    for k in range(cfg.max_layers - 1):
        active = (k < n_layers - 1).unsqueeze(1)
        bound = z0 + cuts[:, k : k + 1] * (z1 - z0)
        layer_id += ((depth > bound) & active).long()
    # soil type per layer, with a mild depth tendency (deeper layers drift the code)
    base = torch.randint(cfg.n_soil_types, (b, cfg.max_layers), generator=g)
    drift = (
        (
            torch.arange(cfg.max_layers).float()
            / cfg.max_layers
            * torch.randn(b, 1, generator=g)
            * 2.0
        )
        .round()
        .long()
    )
    soil_by_layer = (base + drift) % cfg.n_soil_types
    soil_code = torch.gather(soil_by_layer, 1, layer_id)
    return layer_id, soil_code


def _latent_field(
    cfg: GeoPriorConfig,
    depth: torch.Tensor,
    depth_norm: torch.Tensor,
    layer_id: torch.Tensor,
    g: torch.Generator,
) -> torch.Tensor:
    """(B, R, D) piecewise-smooth latent with a shared monotone depth trend.

    ``s = depth_trend + layer_base[layer] + within_walk``:
    - ``within_walk`` — a small per-sqrt-metre random walk reset at each layer
      start (gentle variation within a layer);
    - ``layer_base`` — an independent per-layer signature scaled so a boundary
      jump is ``jump_ratio`` x the adjacent within-layer step (the sharp
      cross-boundary change);
    - ``depth_trend`` — a per-borehole (random-sign, strong) linear function of
      depth shared by all latent dims, so features and target both carry a
      depth trend (real Su rises with effective stress, |corr| ~ 0.8).
    """
    b, r = depth.shape
    d = cfg.latent_dim
    dz = torch.zeros(b, r)
    dz[:, 1:] = (depth[:, 1:] - depth[:, :-1]).clamp(min=0.0)
    steps = (
        cfg.within_step * dz.sqrt().unsqueeze(-1) * torch.randn(b, r, d, generator=g)
    )
    raw = torch.cumsum(steps, dim=1)
    is_start = torch.ones(b, r, dtype=torch.bool)
    is_start[:, 1:] = layer_id[:, 1:] != layer_id[:, :-1]
    idx = torch.arange(r).expand(b, r)
    start_idx = torch.cummax(
        torch.where(is_start, idx, torch.zeros_like(idx)), dim=1
    ).values
    within = raw - torch.gather(raw, 1, start_idx.unsqueeze(-1).expand(b, r, d))

    # boundary jump calibrated to the adjacent within-layer step, not total drift
    adjacent = cfg.within_step * math.sqrt(cfg.spacing_mean)
    ratio = cfg.jump_ratio_min + (cfg.jump_ratio_max - cfg.jump_ratio_min) * torch.rand(
        b, 1, 1, generator=g
    )
    base_scale = ratio * adjacent / math.sqrt(2.0)
    layer_base = torch.randn(b, cfg.max_layers, d, generator=g) * base_scale
    base = torch.gather(layer_base, 1, layer_id.unsqueeze(-1).expand(b, r, d))

    trend_vec = torch.randn(b, 1, d, generator=g) * cfg.depth_trend_scale
    depth_trend = depth_norm.unsqueeze(-1) * trend_vec
    return depth_trend + base + within


def _scm_map(
    cfg: GeoPriorConfig,
    scm_input: torch.Tensor,
    n_feat: int,
    g: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random MLP-SCM: (B, R, in) latent -> (features, target).

    Features are shallow/mid nodes (cheap, upstream); the target is a node from
    the deepest layer (downstream). Returns (feats (B, R, n_feat), target (B, R)).
    """
    b, r, _ = scm_input.shape
    depth = _randint(g, cfg.scm_min_depth, cfg.scm_max_depth)
    widths = [_randint(g, cfg.scm_min_width, cfg.scm_max_width) for _ in range(depth)]
    widths[0] = max(widths[0], n_feat + 1)
    act = _ACTIVATIONS[_randint(g, 0, len(_ACTIVATIONS) - 1)]
    gain = torch.exp(torch.randn(b, 1, 1, generator=g) * 0.5)

    h = scm_input
    nodes = [h]
    for w in widths:
        fan_in = h.shape[-1]
        weight = torch.randn(b, fan_in, w, generator=g) * (gain / math.sqrt(fan_in))
        bias = torch.randn(b, 1, w, generator=g) * 0.2
        h = act(h @ weight + bias)
        nodes.append(h)
    pool = torch.cat(nodes[1:], dim=-1)  # exclude the raw input from the node pool
    pool = (pool - pool.mean(1, keepdim=True)) / (pool.std(1, keepdim=True) + 1e-6)

    n_pool = pool.shape[-1]
    # target from the deepest layer; features from earlier (shallower) nodes
    deep_start = n_pool - widths[-1]
    t_idx = deep_start + torch.randint(widths[-1], (b,), generator=g)
    scores = torch.rand(b, n_pool, generator=g)
    scores[:, deep_start:] = -1.0  # keep features out of the deepest (target) layer
    scores.scatter_(1, t_idx.unsqueeze(1), 2.0)
    f_idx = scores.argsort(dim=1, descending=True)[:, :n_feat]
    feats = pool.gather(2, f_idx.unsqueeze(1).expand(b, r, n_feat))
    target = pool.gather(2, t_idx.view(b, 1, 1).expand(b, r, 1)).squeeze(-1)
    return feats, target


def _generic_features(
    cfg: GeoPriorConfig,
    scm_input: torch.Tensor,
    n_scm_feat: int,
    g: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-MLP-SCM features with a per-column affine + observation noise."""
    b, r, _ = scm_input.shape
    feats, target = _scm_map(cfg, scm_input, n_scm_feat, g)
    scale = torch.exp(torch.randn(b, 1, n_scm_feat, generator=g) * 0.6)
    shift = torch.randn(b, 1, n_scm_feat, generator=g)
    feats = feats * scale + shift
    feats = feats + cfg.obs_noise * scale * torch.randn(b, r, n_scm_feat, generator=g)
    return feats, target


def sample_features(
    cfg: GeoPriorConfig,
    scm_input: torch.Tensor,
    n_scm_feat: int,
    g: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the feature block + target, mixing the generic and geo-realistic priors.

    ``scm_input`` is ``[latent s | depth_norm | location]`` along the last axis.
    Each *table* (batch element) independently uses the realistic cheap-feature
    block with probability ``cfg.p_geo_realistic`` (latent dims 0/1 as the
    soft/signal factors) and the generic random-MLP branch otherwise, so the
    mixing rate is batch-size-independent while every element stays internally
    coherent.
    """
    from geo_pfn.geoprior.realistic import realistic_block

    def _realistic(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sig = 1 if cfg.latent_dim > 1 else 0
        depth_norm = inp[:, :, cfg.latent_dim]
        return realistic_block(
            cfg, inp[:, :, 0], inp[:, :, sig], depth_norm, n_scm_feat, g
        )

    use_real = torch.rand(scm_input.shape[0], generator=g) < cfg.p_geo_realistic
    if not use_real.any():
        return _generic_features(cfg, scm_input, n_scm_feat, g)
    if use_real.all():
        return _realistic(scm_input)
    idx_r = use_real.nonzero(as_tuple=True)[0]
    idx_g = (~use_real).nonzero(as_tuple=True)[0]
    feats_r, target_r = _realistic(scm_input[idx_r])
    feats_g, target_g = _generic_features(cfg, scm_input[idx_g], n_scm_feat, g)
    b, r, _ = scm_input.shape
    feats = torch.empty(b, r, n_scm_feat, dtype=feats_r.dtype)
    target = torch.empty(b, r, dtype=target_r.dtype)
    feats[idx_r], feats[idx_g] = feats_r, feats_g
    target[idx_r], target[idx_g] = target_r, target_g
    return feats, target


def sample_geo_batch(
    cfg: GeoPriorConfig, batch_size: int, generator: torch.Generator
) -> GeoBatch:
    """Sample a batch of synthetic boreholes (on CPU)."""
    g = generator
    n_rows = _randint(g, cfg.min_rows, cfg.max_rows)
    n_feat = _randint(g, cfg.min_features, cfg.max_features)
    b, r = batch_size, n_rows

    depth = _depth_axis(cfg, b, r, g)
    depth_norm = (depth - depth.mean(1, keepdim=True)) / (
        depth.std(1, keepdim=True) + 1e-6
    )
    layer_id, soil_code = _stratigraphy(cfg, depth, g)
    s = _latent_field(cfg, depth, depth_norm, layer_id, g)

    u = torch.randn(b, cfg.latent_dim, generator=g)  # per-borehole location latent
    scm_input = torch.cat(
        [s, depth_norm.unsqueeze(-1), u.unsqueeze(1).expand(b, r, cfg.latent_dim)],
        dim=-1,
    )
    # reserve 4 structural columns (depth, X, Y, soil code); rest are SCM features
    n_scm_feat = max(1, n_feat - 4)
    feats, target = sample_features(cfg, scm_input, n_scm_feat, g)

    coords = u[:, :2].unsqueeze(1).expand(b, r, 2)
    x = torch.cat(
        [depth.unsqueeze(-1), coords, soil_code.float().unsqueeze(-1), feats], dim=-1
    )

    # block column missingness (whole column absent for the whole borehole)
    if cfg.block_missing_prob > 0 and n_scm_feat > 1:
        drop = torch.rand(b, 1, n_scm_feat, generator=g) < cfg.block_missing_prob
        keep_one = drop.all(-1, keepdim=True) & (torch.arange(n_scm_feat) == 0).view(
            1, 1, -1
        )
        drop = drop & ~keep_one  # never blank every feature column
        feat_block = x[:, :, 4:].masked_fill(drop.expand(b, r, n_scm_feat), math.nan)
        x = torch.cat([x[:, :, :4], feat_block], dim=-1)

    train_frac = cfg.min_train_frac + (cfg.max_train_frac - cfg.min_train_frac) * float(
        torch.rand(1, generator=g).item()
    )
    train_size = min(max(int(n_rows * train_frac), 2), n_rows - 1)

    return GeoBatch(
        x=x,
        y=target,
        depth=depth,
        layer_id=layer_id,
        soil_code=soil_code.long(),
        train_size=train_size,
    )
