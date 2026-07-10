"""Calibration metrics for bar-distribution mixture predictions.

A geo-PFN ensemble prediction for one query row is a mixture of ``E`` histograms:
component ``e`` places probabilities ``probs[e]`` over the fixed normalized bin
edges ``z_edges``, mapped to real units by that draw's context affine
``v = z * ctx_std[e] + ctx_mean[e]``. Every metric here is computed exactly
(closed form) on that piecewise-linear mixture CDF — no sampling:

- ``mixture_stats``     mean / std of the mixture (law of total variance)
- ``mixture_quantiles`` inverse-CDF at arbitrary levels (central intervals)
- ``mixture_crps``      continuous ranked probability score
- ``mixture_nll``       negative log density at the truth (+ outside-support flag)

These are what the sparse-regime uncertainty evaluation is built on: coverage of
central intervals vs nominal (calibration), CRPS/NLL (calibrated + sharp), and
whether predicted std tracks actual error.
"""

from __future__ import annotations

import numpy as np


def _component_edges(
    ctx_mean: np.ndarray, ctx_std: np.ndarray, z_edges: np.ndarray
) -> np.ndarray:
    """Real-unit bin edges per component, shape (E, B+1)."""
    return z_edges[None, :] * ctx_std[:, None] + ctx_mean[:, None]


def mixture_stats(
    probs: np.ndarray, ctx_mean: np.ndarray, ctx_std: np.ndarray, z_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the mixture per query row.

    ``probs`` is (E, Q, B); returns two (Q,) arrays. Uses bin centers (the same
    approximation as the model's own ``mean_std``) and the law of total variance.
    """
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
    z_width = z_edges[1] - z_edges[0]
    # per-component moments in z space
    mu_z = probs @ z_centers  # (E, Q)
    var_z = (probs * (z_centers[None, None, :] - mu_z[..., None]) ** 2).sum(-1)
    var_z = var_z + z_width**2 / 12.0  # within-bin (uniform) variance
    # to real units
    mu = mu_z * ctx_std[:, None] + ctx_mean[:, None]  # (E, Q)
    var = var_z * (ctx_std[:, None] ** 2)
    mean = mu.mean(0)
    second = (var + mu**2).mean(0)
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def _mixture_cdf_at(
    v: np.ndarray,
    probs: np.ndarray,
    edges: np.ndarray,
    cum: np.ndarray,
) -> np.ndarray:
    """Mixture CDF evaluated at points ``v`` (Q, K) -> (Q, K).

    ``edges`` is (E, B+1) real-unit component edges, ``cum`` is (E, Q, B+1) the
    per-component cumulative probabilities (0 .. 1 along the last axis).
    """
    e_n, q_n, _ = cum.shape
    out = np.zeros((q_n, v.shape[1]))
    for e in range(e_n):
        idx = np.searchsorted(edges[e], v, side="right")  # (Q, K) in 0..B+1
        idx = np.clip(idx, 1, edges.shape[1] - 1)
        lo, hi = edges[e][idx - 1], edges[e][idx]
        frac = np.clip((v - lo) / np.maximum(hi - lo, 1e-30), 0.0, 1.0)
        c_lo = np.take_along_axis(cum[e], idx - 1, axis=1)
        c_hi = np.take_along_axis(cum[e], idx, axis=1)
        comp = c_lo + frac * (c_hi - c_lo)
        comp = np.where(v < edges[e][0], 0.0, comp)
        comp = np.where(v >= edges[e][-1], 1.0, comp)
        out += comp
    return out / e_n


def _prepare(
    probs: np.ndarray, ctx_mean: np.ndarray, ctx_std: np.ndarray, z_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Component real-unit edges (E, B+1) and cumulative probs (E, Q, B+1)."""
    edges = _component_edges(ctx_mean, ctx_std, z_edges)
    e_n, q_n, b = probs.shape
    cum = np.zeros((e_n, q_n, b + 1))
    np.cumsum(probs, axis=-1, out=cum[:, :, 1:])
    cum[:, :, -1] = 1.0  # guard against fp drift
    return edges, cum


def mixture_quantiles(
    probs: np.ndarray,
    ctx_mean: np.ndarray,
    ctx_std: np.ndarray,
    z_edges: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """Quantiles of the mixture per query row: returns (Q, len(levels)).

    Inverts the piecewise-linear mixture CDF by bisection over the support (exact
    to the tolerance below, default ~1e-4 of the support width).
    """
    edges, cum = _prepare(probs, ctx_mean, ctx_std, z_edges)
    q_n = probs.shape[1]
    lo = np.full((q_n, len(levels)), edges[:, 0].min())
    hi = np.full((q_n, len(levels)), edges[:, -1].max())
    target = np.broadcast_to(levels[None, :], (q_n, len(levels)))
    for _ in range(40):  # 2^-40 of support width
        mid = 0.5 * (lo + hi)
        f = _mixture_cdf_at(mid, probs, edges, cum)
        lo = np.where(f < target, mid, lo)
        hi = np.where(f < target, hi, mid)
    return 0.5 * (lo + hi)


def mixture_crps(
    probs: np.ndarray,
    ctx_mean: np.ndarray,
    ctx_std: np.ndarray,
    z_edges: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Exact CRPS per query row: integral of (F(x) - 1{x >= y})^2.

    The mixture CDF is piecewise linear on the union of all component edges plus
    ``y``; on each segment the integrand is quadratic, integrated in closed form.
    Returns (Q,).
    """
    edges, cum = _prepare(probs, ctx_mean, ctx_std, z_edges)
    q_n = probs.shape[1]
    # breakpoints: all component edges (shared across rows) + per-row y
    base = np.sort(np.unique(edges.reshape(-1)))
    pts = np.sort(
        np.concatenate(
            [np.broadcast_to(base[None, :], (q_n, base.size)), y_true[:, None]], axis=1
        ),
        axis=1,
    )  # (Q, K)
    f = _mixture_cdf_at(pts, probs, edges, cum)  # (Q, K)
    a, b = pts[:, :-1], pts[:, 1:]
    fa, fb = f[:, :-1], f[:, 1:]
    ind = (a >= y_true[:, None]).astype(float)  # indicator constant per segment
    da, db = fa - ind, fb - fa
    seg = (b - a) * (da**2 + da * db + db**2 / 3.0)
    return seg.sum(1)


def mixture_nll(
    probs: np.ndarray,
    ctx_mean: np.ndarray,
    ctx_std: np.ndarray,
    z_edges: np.ndarray,
    y_true: np.ndarray,
    floor: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Negative log mixture density at the truth, and an outside-support flag.

    Density is the probability of the containing bin divided by its real-unit
    width, averaged over components (zero for components whose support excludes
    ``y``). ``outside`` marks rows where *every* component puts zero density —
    a direct measure of catastrophic over-confidence. NLL is floored so those
    rows stay finite; report ``outside`` separately. Returns ((Q,), (Q,) bool).
    """
    edges = _component_edges(ctx_mean, ctx_std, z_edges)
    e_n, q_n, b = probs.shape
    z_width = z_edges[1] - z_edges[0]
    dens = np.zeros(q_n)
    for e in range(e_n):
        idx = np.searchsorted(edges[e], y_true, side="right") - 1  # (Q,)
        inside = (idx >= 0) & (idx < b)
        safe = np.clip(idx, 0, b - 1)
        width = z_width * ctx_std[e]
        dens += np.where(inside, probs[e, np.arange(q_n), safe] / width, 0.0)
    dens /= e_n
    outside = dens <= 0.0
    return -np.log(np.maximum(dens, floor)), outside
