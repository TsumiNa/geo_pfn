"""Tests for calibration metrics: closed forms vs Monte-Carlo estimates."""

from __future__ import annotations

import numpy as np

from geo_pfn.geopfn.calibration import (
    mixture_crps,
    mixture_nll,
    mixture_quantiles,
    mixture_stats,
)


def _random_mixture(rng, e_n=4, q_n=6, b=32):
    z_edges = np.linspace(-4.0, 4.0, b + 1)
    logits = rng.normal(size=(e_n, q_n, b)) * 2.0
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    ctx_mean = rng.normal(size=e_n) * 10.0 + 50.0
    ctx_std = rng.uniform(5.0, 20.0, size=e_n)
    return probs, ctx_mean, ctx_std, z_edges


def _sample(rng, probs, ctx_mean, ctx_std, z_edges, q, n):
    """Draw n samples from query row q's mixture."""
    e_n, _, b = probs.shape
    comp = rng.integers(0, e_n, size=n)
    out = np.empty(n)
    for e in range(e_n):
        m = comp == e
        if not m.any():
            continue
        bins = rng.choice(b, size=m.sum(), p=probs[e, q])
        u = rng.uniform(size=m.sum())
        z = z_edges[bins] + u * (z_edges[1] - z_edges[0])
        out[m] = z * ctx_std[e] + ctx_mean[e]
    return out


def test_stats_match_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    probs, cm, cs, ze = _random_mixture(rng)
    mean, std = mixture_stats(probs, cm, cs, ze)
    for q in range(probs.shape[1]):
        s = _sample(rng, probs, cm, cs, ze, q, 200_000)
        assert abs(mean[q] - s.mean()) < 0.15
        assert abs(std[q] - s.std()) < 0.15


def test_crps_matches_monte_carlo() -> None:
    rng = np.random.default_rng(1)
    probs, cm, cs, ze = _random_mixture(rng, q_n=4)
    y = rng.normal(50.0, 25.0, size=4)
    crps = mixture_crps(probs, cm, cs, ze, y)
    for q in range(4):
        s1 = _sample(rng, probs, cm, cs, ze, q, 100_000)
        s2 = _sample(rng, probs, cm, cs, ze, q, 100_000)
        mc = np.abs(s1 - y[q]).mean() - 0.5 * np.abs(s1 - s2).mean()
        assert abs(crps[q] - mc) < 0.2, (q, crps[q], mc)


def test_crps_point_mass_limit_is_abs_error() -> None:
    # a single near-degenerate bin -> CRPS ~ |y - center|
    b = 64
    ze = np.linspace(-4, 4, b + 1)
    probs = np.zeros((1, 1, b))
    probs[0, 0, b // 2] = 1.0  # bin straddling z=0
    cm, cs = np.array([100.0]), np.array([0.01])  # ~point mass at 100
    y = np.array([107.0])
    crps = mixture_crps(probs, cm, cs, ze, y)
    assert abs(crps[0] - 7.0) < 0.01


def test_quantiles_invert_cdf() -> None:
    rng = np.random.default_rng(2)
    probs, cm, cs, ze = _random_mixture(rng, q_n=3)
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    qs = mixture_quantiles(probs, cm, cs, ze, levels)
    # exactness criterion: the mixture CDF at each computed quantile is the level
    from geo_pfn.geopfn.calibration import _mixture_cdf_at, _prepare

    edges, cum = _prepare(probs, cm, cs, ze)
    f = _mixture_cdf_at(qs, probs, edges, cum)
    assert np.allclose(f, levels[None, :], atol=1e-4)
    # and it agrees with empirical quantiles up to MC noise
    for q in range(3):
        s = _sample(rng, probs, cm, cs, ze, q, 200_000)
        emp = np.quantile(s, levels)
        assert np.allclose(qs[q], emp, atol=0.8), (qs[q], emp)


def test_nll_uniform_single_bin() -> None:
    b = 8
    ze = np.linspace(-4, 4, b + 1)
    probs = np.zeros((1, 1, b))
    probs[0, 0, 3] = 1.0
    cm, cs = np.array([0.0]), np.array([1.0])
    width = ze[1] - ze[0]  # real width = z width * std(=1)
    y_in = np.array([ze[3] + width / 2])
    nll, outside = mixture_nll(probs, cm, cs, ze, y_in)
    assert not outside[0]
    assert abs(nll[0] - np.log(width)) < 1e-9
    y_out = np.array([100.0])
    nll2, outside2 = mixture_nll(probs, cm, cs, ze, y_out)
    assert outside2[0] and np.isfinite(nll2[0])
