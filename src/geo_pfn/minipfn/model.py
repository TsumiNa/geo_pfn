"""Mini-PFN: a per-cell tokenized transformer for tabular in-context learning.

Follows the TabPFN v2 layout at small scale: every cell of the (train + test)
table is a token; blocks alternate attention across the feature axis (within a
row) and across the sample axis (within a column, keys/values from train rows
only, so test predictions are independent of other test rows); the target is
an extra token column and predictions are read from it at the test rows.

Missing values (NaN) are first-class inputs: each cell is encoded as
``[z-scored value (0 where missing), missing flag]``, mirroring TabPFN's
NaN-indicator + train-mean-imputation encoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as func

from geo_pfn.minipfn.config import ModelConfig


class _Attention(nn.Module):
    """Multi-head attention with an explicit key/value sequence."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        n, len_q, d = x_q.shape
        heads = self.n_heads
        q = self.q_proj(x_q).view(n, len_q, heads, d // heads).transpose(1, 2)
        k = self.k_proj(x_kv).view(n, -1, heads, d // heads).transpose(1, 2)
        v = self.v_proj(x_kv).view(n, -1, heads, d // heads).transpose(1, 2)
        out = func.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(n, len_q, d))


class _Block(nn.Module):
    """Pre-norm block: feature-axis attention, sample-axis attention, MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.feature_attn = _Attention(d, config.n_heads)
        self.row_attn = _Attention(d, config.n_heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm_feat = nn.LayerNorm(d)
        self.norm_row = nn.LayerNorm(d)
        self.norm_mlp = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor, train_size: int) -> torch.Tensor:
        b, r, c, d = tokens.shape
        h = self.norm_feat(tokens).reshape(b * r, c, d)
        tokens = tokens + self.feature_attn(h, h).view(b, r, c, d)

        h = self.norm_row(tokens).permute(0, 2, 1, 3).reshape(b * c, r, d)
        attn = self.row_attn(h, h[:, :train_size])
        tokens = tokens + attn.view(b, c, r, d).permute(0, 2, 1, 3)

        return tokens + self.mlp(self.norm_mlp(tokens))


class MiniPFN(nn.Module):
    """In-context tabular classifier over a (train + test) table with NaN support."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.cell_encoder = nn.Linear(2, d)
        self.feature_emb_proj = nn.Linear(config.feature_emb_dim, d, bias=False)
        self.y_encoder = nn.Linear(config.max_classes, d)
        self.y_mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.y_col_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layers))
        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.max_classes)
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, train_size: int
    ) -> torch.Tensor:
        """Predict class logits for the test rows.

        Args:
            x: (B, R, F) features for train and test rows; NaN marks missing cells.
            y: (B, R) integer labels; values at rows >= train_size are ignored.
            train_size: number of leading context rows.

        Returns:
            (B, R - train_size, max_classes) logits.
        """
        b, r, n_feat = x.shape
        if not 0 < train_size < r:
            raise ValueError(
                "train_size must leave at least one train and one test row"
            )

        # per-column z-normalization from train-row statistics, NaN-aware
        missing = torch.isnan(x)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        train_x = x[:, :train_size]
        train_obs = ~missing[:, :train_size]
        count = train_obs.sum(dim=1).clamp(min=1)  # (B, F)
        mean = torch.where(train_obs, train_x, zero).sum(dim=1) / count
        dev = train_x - mean.unsqueeze(1)
        var = torch.where(train_obs, dev * dev, zero).sum(dim=1) / count
        std = torch.where(count >= 2, (var + 1e-8).sqrt(), torch.ones_like(var)).clamp(
            min=1e-6
        )
        xn = ((x - mean.unsqueeze(1)) / std.unsqueeze(1)).clamp(-10.0, 10.0)
        xn = torch.where(missing, zero, xn)

        cells = self.cell_encoder(torch.stack([xn, missing.to(x.dtype)], dim=-1))
        cells = cells + self.feature_emb_proj(
            self._feature_codes(b, n_feat, x.device)
        ).unsqueeze(1)

        y_onehot = func.one_hot(
            y.clamp(0, self.config.max_classes - 1), self.config.max_classes
        )
        is_test = (torch.arange(r, device=x.device) >= train_size).view(1, r, 1)
        y_tok = torch.where(
            is_test, self.y_mask_emb, self.y_encoder(y_onehot.to(x.dtype))
        )
        y_tok = y_tok + self.y_col_emb

        tokens = torch.cat([cells, y_tok.unsqueeze(2)], dim=2)
        for block in self.blocks:
            tokens = block(tokens, train_size)

        return self.head(self.head_norm(tokens[:, train_size:, -1]))

    def _feature_codes(
        self, batch_size: int, n_feat: int, device: torch.device
    ) -> torch.Tensor:
        """Random per-column identity codes; resampled while training, fixed in eval."""
        shape = (batch_size, n_feat, self.config.feature_emb_dim)
        if self.training:
            return torch.randn(shape, device=device)
        generator = torch.Generator().manual_seed(0)
        return torch.randn(shape, generator=generator).to(device)


def mask_class_logits(logits: torch.Tensor, n_classes: torch.Tensor) -> torch.Tensor:
    """Fill logits of class indices >= ``n_classes[b]`` with a large negative value."""
    class_idx = torch.arange(logits.shape[-1], device=logits.device)
    invalid = class_idx.view(1, 1, -1) >= n_classes.view(-1, 1, 1)
    return logits.masked_fill(invalid, -1e9)
