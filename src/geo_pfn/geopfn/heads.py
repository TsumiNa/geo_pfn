"""Bar-distribution regression head (Riemann histogram over the target).

The head predicts a categorical distribution over fixed bins spanning the
(context-normalized) target range. This gives, in one forward pass, both a
point estimate (the distribution's mean) and calibrated uncertainty (its std) —
the "value + uncertainty" the geo-PFN vision needs, and the standard way TabPFN
and TabICL do tabular regression.
"""

from __future__ import annotations

import torch
from torch import nn


class BarDistribution(nn.Module):
    """Fixed-bin distribution over a normalized target range ``[-half, half]``."""

    def __init__(self, n_bins: int = 64, half_range: float = 4.0) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if half_range <= 0:
            raise ValueError("half_range must be positive")
        edges = torch.linspace(-half_range, half_range, n_bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2)
        self.n_bins = n_bins

    def to_bins(self, z: torch.Tensor) -> torch.Tensor:
        """Bin index of normalized targets ``z`` (clamped into range)."""
        idx = torch.bucketize(z, self.edges[1:-1].contiguous())
        return idx.clamp(0, self.n_bins - 1)

    def loss(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of bin logits against the true bin of ``z``."""
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.n_bins), self.to_bins(z).reshape(-1)
        )

    def mean_std(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Distribution mean and std (in normalized space) from bin logits."""
        p = torch.softmax(logits, dim=-1)
        centers = self.centers.to(p.dtype)
        mean = (p * centers).sum(-1)
        var = (p * (centers - mean.unsqueeze(-1)) ** 2).sum(-1)
        return mean, var.clamp(min=0).sqrt()
