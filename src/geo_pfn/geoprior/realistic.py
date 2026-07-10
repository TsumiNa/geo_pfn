"""Geo-realistic cheap-feature block (docs/geo-scm-design.md §10).

The generic random-MLP branch (``prior._scm_map``) makes every synthetic feature
informative and roughly independent. The real Haneda cheap-geotech features are
the opposite — a tightly collinear cluster whose dominant axis is redundant with
depth and carries almost no Su signal, while a faint low-variance contrast among
the features carries what little extra signal there is (see ``fit_stats`` and the
design doc for the R^2 / PCA evidence). Training a PFN on the generic branch alone
teaches it to trust high-variance feature directions, which is exactly wrong for
this data and is why feeding geo-PFN the real LCSG columns *hurt* it.

This module samples a cheap-feature block that reproduces both facts, driven by
two smooth latent factors already present in the geo-SCM field:

- ``soft`` — the dominant "consistency / water-state" axis. Loads the cluster
  along the real PC1 direction (Wn/LL/PL/e one way, rho_t the other), is
  depth-trended (so redundant with depth), and contributes almost nothing to the
  target. This is the 76%-of-variance, ~0-signal axis.
- ``signal`` — a small-amplitude feature contrast that *does* drive a weak Su
  term. Low feature variance, but the source of the real "+0.12 R^2 beyond depth".

Each feature is then mapped through the real column's empirical quantile function,
so the marginals (near-constant Gs, skewed grain-size, etc.) match too. Per-site
random jitter of the loadings, the contrast direction, and the target coefficients
keeps this a *family* of laws — the model still has to read each site's coupling
from context rather than memorise one mapping.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import torch

from geo_pfn.geoprior.config import GeoPriorConfig

_STATS_PATH = Path(__file__).with_name("haneda_stats.json")
_SQRT2 = math.sqrt(2.0)


@lru_cache(maxsize=1)
def load_stats() -> dict:
    """Load the fitted Haneda marginals/coupling (see ``fit_stats``)."""
    return json.loads(_STATS_PATH.read_text())


def _standardize(v: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-std along the row axis (dim 1), per batch element."""
    mean = v.mean(dim=1, keepdim=True)
    std = v.std(dim=1, keepdim=True).clamp(min=1e-6)
    return (v - mean) / std


def _normal_cdf(g: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(g / _SQRT2))


def _quantile_map(u: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Map u in [0, 1] through an empirical quantile grid ``q`` (increasing)."""
    n = q.numel()
    p = torch.linspace(0.0, 1.0, n)
    idx = torch.searchsorted(p, u.clamp(0.0, 1.0), right=True).clamp(1, n - 1)
    x0, x1 = p[idx - 1], p[idx]
    y0, y1 = q[idx - 1], q[idx]
    t = ((u - x0) / (x1 - x0).clamp(min=1e-12)).clamp(0.0, 1.0)
    return y0 + t * (y1 - y0)


def realistic_block(
    cfg: GeoPriorConfig,
    soft: torch.Tensor,
    signal: torch.Tensor,
    depth_norm: torch.Tensor,
    n_feat: int,
    g: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a geo-realistic (features, target) pair.

    ``soft``/``signal``/``depth_norm`` are (B, R) smooth latent factors from the
    geo-SCM field. Returns ``feats`` (B, R, n_feat) in real-ish units and
    ``target`` (B, R). The first up-to-6 columns are the cheap-geotech cluster
    (real PC1 loadings + marginals), the next up-to-4 are grain-size (zero-inflated
    real marginals), any remainder are generic normal columns for breadth.
    """
    stats = load_stats()
    b, r = soft.shape
    soft = _standardize(soft).unsqueeze(-1)  # (B, R, 1)
    signal = _standardize(signal).unsqueeze(-1)

    cheap = stats["cheap_columns"]
    pc1 = torch.tensor(stats["cheap_pc1"], dtype=torch.float32)  # (6,)
    n_cheap = min(n_feat, len(cheap))

    cols: list[torch.Tensor] = []
    # --- cheap-geotech collinear cluster -------------------------------------
    if n_cheap > 0:
        # per-site jitter of the cluster orientation and of the faint contrast
        load = pc1[:n_cheap].view(1, 1, n_cheap)
        load = load + cfg.realistic_load_jitter * torch.randn(
            b, 1, n_cheap, generator=g
        )
        contrast = torch.randn(b, 1, n_cheap, generator=g)  # signal-axis loadings
        contrast = contrast - contrast.mean(-1, keepdim=True)  # ~orthogonal to level
        idio = torch.randn(b, r, n_cheap, generator=g)
        gauss = (
            load * soft
            + cfg.realistic_signal_amp * contrast * signal
            + cfg.realistic_idio_amp * idio
        )
        gauss = _standardize(gauss)
        u = _normal_cdf(gauss)
        for j in range(n_cheap):
            q = torch.tensor(stats["cheap_quantiles"][cheap[j]], dtype=torch.float32)
            cols.append(_quantile_map(u[:, :, j], q))

    # --- grain-size: zero-inflated real marginals, weakly driven -------------
    grain = stats["grain_columns"]
    n_grain = min(max(0, n_feat - len(cheap)), len(grain))
    if n_grain > 0:
        gload = torch.randn(b, 1, n_grain, generator=g)
        ggauss = _standardize(
            gload * signal
            + cfg.realistic_idio_amp * torch.randn(b, r, n_grain, generator=g)
        )
        gu = _normal_cdf(ggauss)
        for j in range(n_grain):
            gs = stats["grain"][grain[j]]
            q = torch.tensor(gs["pos_quantiles"], dtype=torch.float32)
            zero_frac = gs["zero_frac"]
            # lower zero_frac fraction of the CDF collapses to 0 (zero-inflation)
            scaled = ((gu[:, :, j] - zero_frac) / max(1e-6, 1.0 - zero_frac)).clamp(
                0.0, 1.0
            )
            val = _quantile_map(scaled, q)
            cols.append(
                torch.where(gu[:, :, j] < zero_frac, torch.zeros_like(val), val)
            )

    # --- generic breadth columns (unknown extra features) --------------------
    n_generic = n_feat - len(cols)
    if n_generic > 0:
        w_soft = torch.randn(b, 1, n_generic, generator=g)
        w_sig = torch.randn(b, 1, n_generic, generator=g)
        extra = (
            w_soft * soft + w_sig * signal + torch.randn(b, r, n_generic, generator=g)
        )
        for j in range(n_generic):
            cols.append(extra[:, :, j])

    feats = torch.stack(cols, dim=-1)  # (B, R, n_feat)

    # --- target: depth-dominated, ~no soft signal, weak signal-axis term -----
    coef = cfg.realistic_target_coefs
    jit = 1.0 + cfg.realistic_coef_jitter * torch.randn(b, 1, 3, generator=g)
    noise = torch.randn(b, r, generator=g)
    target = (
        coef[0] * jit[:, :, 0] * depth_norm
        + coef[1] * jit[:, :, 1] * soft.squeeze(-1)
        + coef[2] * jit[:, :, 2] * signal.squeeze(-1)
        + cfg.realistic_target_noise * noise
    )
    return feats, target
