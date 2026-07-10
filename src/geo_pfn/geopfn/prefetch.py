"""Background multi-process prefetcher for geo-SCM training batches.

The prior sampler (``sample_geo_site_batch``) is a CPU-bound serial Python loop,
so a single-process training loop leaves the GPU idle between steps (on an A100 a
2M model sits at ~20% utilisation, starved for data). This spreads sampling across
worker processes that keep a bounded queue of ready batches full, so the GPU never
waits — turning a CPU-bound loop into a GPU-bound one.

Opt-in via ``train --num-workers N``. ``N=0`` keeps the deterministic
single-process path (used for the reproducible A/B). With workers > 0 the arrival
order of batches across workers is non-deterministic, so a run is not bit-for-bit
reproducible — acceptable for PFN training, which is already an average over an
endless stream of random synthetic tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.multiprocessing as mp

from geo_pfn.geoprior.prior import GeoBatch

Sampler = Callable[[Any, int, torch.Generator], GeoBatch]


def _worker(
    sampler_fn: Sampler, prior_cfg: Any, batch_size: int, seed: int, queue: mp.Queue
) -> None:
    """Endlessly sample batches into ``queue`` (blocks when the queue is full).

    Each worker is pinned to a single intra-op thread: sampling parallelism comes
    from having many worker *processes*, so letting each also open a full OpenMP
    threadpool would oversubscribe the CPU (N_workers x N_cores threads) and thrash
    any concurrent CPU work. Keeps the box usable while the GPU stays fed.
    ``sampler_fn`` is any module-level ``(cfg, batch_size, generator) -> GeoBatch``.
    """
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)
    while True:
        queue.put(sampler_fn(prior_cfg, batch_size, generator))


class BatchPrefetcher:
    """Pool of spawn workers sampling geo-SCM batches into a bounded queue.

    Each worker owns a distinct generator stream (``base_seed`` offset by worker
    id), so the pooled output stays diverse. ``get`` pops the next ready batch;
    the bounded queue applies backpressure so at most ``prefetch`` batches (plus
    one per worker in flight) are held in shared memory at once.
    """

    def __init__(
        self,
        sampler_fn: Sampler,
        prior_cfg: Any,
        batch_size: int,
        base_seed: int,
        num_workers: int,
        prefetch: int | None = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        ctx = mp.get_context("spawn")
        self._queue: mp.Queue = ctx.Queue(maxsize=prefetch or 2 * num_workers)
        self._procs = []
        for wid in range(num_workers):
            proc = ctx.Process(
                target=_worker,
                args=(
                    sampler_fn,
                    prior_cfg,
                    batch_size,
                    base_seed * 100_003 + wid,
                    self._queue,
                ),
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)

    def get(self) -> GeoBatch:
        return self._queue.get()

    def close(self) -> None:
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            proc.join(timeout=2)

    def __enter__(self) -> BatchPrefetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
