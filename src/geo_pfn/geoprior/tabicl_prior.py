"""Train geo-PFN on TabICL's SCM prior (SCM-swap ablation).

This lets us feed our architecture TabICL's *own* synthetic prior (its mix of
MLP-SCM and tree-SCM tabular tasks), so that geo-PFN-on-TabICL-SCM vs
geo-PFN-on-our-geo-SCM isolates whether the accuracy gap to TabICL is the prior
or the architecture.

TabICL's ``PriorDataset`` natively produces *classification* tables (its SCM's
continuous output is discretised by ``Reg2Cls``). We keep it continuous — the
regression target our bar-distribution head needs — by forcing ``num_classes=0``
(``Reg2Cls`` then only standardises y, never bins it) and relaxing the
class-oriented ``sanity_check``. Everything else (the SCMs, feature processing,
padding to ``max_features``, group structure) is TabICL's, unchanged.

Requires the overlay deps: ``uv run --with tabicl --with xgboost``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geo_pfn.geoprior.prior import GeoBatch

_PATCHED = False


def _ensure_regression_patch() -> None:
    """Monkeypatch TabICL's SCM prior to emit continuous (regression) targets.

    Applied lazily, in-process — so it also takes effect inside prefetch spawn
    workers (which import this module) without pickling a closure.
    """
    global _PATCHED
    if _PATCHED:
        return
    from tabicl.prior._dataset import SCMPrior

    original = SCMPrior.generate_dataset

    def _regression_generate(self, params):  # type: ignore[no-untyped-def]
        return original(self, {**params, "num_classes": 0})

    SCMPrior.generate_dataset = _regression_generate  # type: ignore[assignment]
    SCMPrior.sanity_check = staticmethod(  # type: ignore[assignment]
        lambda X, y, train_size, **kw: bool(
            not torch.isnan(y).any() and not torch.isnan(X).any()
        )
    )
    _PATCHED = True


@dataclass(kw_only=True)
class TabiclPriorConfig:
    """Shape of the TabICL-prior tables fed to geo-PFN (scaled to our setting)."""

    min_features: int = 2
    max_features: int = 24
    min_seq_len: int = 64
    max_seq_len: int = 384
    min_train_size: float = 0.3
    max_train_size: float = 0.7
    prior_type: str = "mix_scm"  # TabICL's MLP-SCM + tree-SCM mixture


def sample_tabicl_batch(
    cfg: TabiclPriorConfig, batch_size: int, generator: torch.Generator
) -> GeoBatch:
    """Sample one batch of TabICL-SCM regression tables as a ``GeoBatch``.

    ``x`` is (B, R, F) features (padded to ``max_features``), ``y`` is (B, R) the
    continuous SCM target, and ``context_mask`` marks the first ``train_size`` rows
    of each table as context. The geo-specific fields are unused by training and
    filled with placeholders.
    """
    _ensure_regression_patch()
    from tabicl.prior._dataset import PriorDataset

    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    torch.manual_seed(seed)
    import numpy as np

    np.random.seed(seed)

    ds = PriorDataset(
        batch_size=batch_size,
        min_features=cfg.min_features,
        max_features=cfg.max_features,
        min_seq_len=cfg.min_seq_len,
        max_seq_len=cfg.max_seq_len,
        min_train_size=cfg.min_train_size,
        max_train_size=cfg.max_train_size,
        prior_type=cfg.prior_type,
        device="cpu",
        n_jobs=1,  # parallelism comes from geo-PFN's prefetch pool, not joblib
    )
    x, y, _d, _seq_lens, train_sizes = ds.get_batch(batch_size)
    b, r, _f = x.shape
    ctx = torch.zeros(b, r, dtype=torch.bool)
    for i in range(b):
        ctx[i, : int(train_sizes[i])] = True
    return GeoBatch(
        x=x.float(),
        y=y.float(),
        depth=torch.zeros(b, r),
        layer_id=torch.zeros(b, r, dtype=torch.long),
        soil_code=torch.zeros(b, r, dtype=torch.long),
        train_size=-1,
        context_mask=ctx,
    )
