"""Test the background batch prefetcher."""

from __future__ import annotations

import torch

from geo_pfn.geopfn.prefetch import BatchPrefetcher
from geo_pfn.geoprior.config import GeoPriorConfig


def test_prefetcher_yields_valid_batches() -> None:
    cfg = GeoPriorConfig()
    pf = BatchPrefetcher(cfg, batch_size=4, base_seed=0, num_workers=2, prefetch=4)
    try:
        for _ in range(3):
            batch = pf.get()
            assert batch.x.shape[0] == 4
            assert batch.context_mask.shape == batch.y.shape
            assert torch.isfinite(batch.y).all()
            assert batch.context_mask.any() and (~batch.context_mask).any()
    finally:
        pf.close()
