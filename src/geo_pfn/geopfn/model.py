"""Two-stage geo-PFN (docs/geo-scm-design.md §9).

Three stages that make "row similarity" a first-class object, unlike MiniPFN's
per-cell alternating attention:

1. **Column encoder** — each cell attends along the sample axis within its own
   column (keys from context rows only), so a cell embedding knows where it sits
   in that feature's distribution.
2. **Row compression** — a CLS token attends over a row's feature cells,
   compressing each row into one vector.
3. **ICL over rows** — query row vectors attend to context row vectors (which
   carry the target), a learned metric / soft kNN. This is where borehole
   similarity is exploited.

Targets are read from a bar-distribution head (value + uncertainty). Context is a
per-row boolean mask (not a contiguous split), so the site prior's designed
anchor scenario — dense neighbours + sparse targets — is used directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as func

from geo_pfn.geopfn.heads import BarDistribution


@dataclass(kw_only=True)
class GeoPFNConfig:
    """Architecture of the two-stage geo-PFN."""

    d_model: int = 160
    n_heads: int = 8
    col_layers: int = 2
    row_layers: int = 4
    feature_emb_dim: int = 48
    n_bins: int = 64
    half_range: float = 4.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.col_layers, self.row_layers) < 1:
            raise ValueError("col_layers and row_layers must be >= 1")


class _Attention(nn.Module):
    """Multi-head attention with an optional boolean key mask (True = usable key)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, lq, d = x_q.shape
        h = self.n_heads
        q = self.q_proj(x_q).view(n, lq, h, d // h).transpose(1, 2)
        k = self.k_proj(x_kv).view(n, -1, h, d // h).transpose(1, 2)
        v = self.v_proj(x_kv).view(n, -1, h, d // h).transpose(1, 2)
        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask.view(n, 1, 1, -1)  # (n, 1, 1, L_kv), broadcast
        out = func.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(n, lq, d))


class _Block(nn.Module):
    """Pre-norm self-attention + MLP block with an optional key mask."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = _Attention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_m = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm_a(x)
        x = x + self.attn(h, h, key_mask=key_mask)
        return x + self.mlp(self.norm_m(x))


class GeoPFN(nn.Module):
    """Two-stage in-context regressor over a (context + query) borehole table."""

    def __init__(self, config: GeoPFNConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.cell_encoder = nn.Linear(2, d)
        self.feature_emb_proj = nn.Linear(config.feature_emb_dim, d, bias=False)
        self.col_blocks = nn.ModuleList(
            _Block(d, config.n_heads) for _ in range(config.col_layers)
        )
        self.row_cls = nn.Parameter(torch.randn(d) * 0.02)
        self.compress = _Attention(d, config.n_heads)
        self.compress_norm = nn.LayerNorm(d)
        self.y_encoder = nn.Linear(1, d)
        self.y_mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.icl_blocks = nn.ModuleList(
            _Block(d, config.n_heads) for _ in range(config.row_layers)
        )
        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.n_bins)
        )
        self.bar = BarDistribution(config.n_bins, config.half_range)
        self._eval_codes: dict[tuple[int, int, str], torch.Tensor] = {}

    def forward(
        self, x: torch.Tensor, y_norm: torch.Tensor, context_mask: torch.Tensor
    ) -> torch.Tensor:
        """Bin logits for every row. Args: x (B,R,F) NaN=missing; y_norm (B,R)
        context-normalized target; context_mask (B,R) bool. Returns (B,R,n_bins)."""
        b, r, n_feat = x.shape
        missing = torch.isnan(x)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        ctx = context_mask.unsqueeze(-1)  # (B, R, 1)
        obs = ctx & ~missing
        count = obs.sum(dim=1).clamp(min=1)  # (B, F)
        mean = torch.where(obs, x, zero).sum(dim=1) / count
        dev = x - mean.unsqueeze(1)
        var = torch.where(obs, dev * dev, zero).sum(dim=1) / count
        std = torch.where(count >= 2, (var + 1e-8).sqrt(), torch.ones_like(var)).clamp(
            min=1e-6
        )
        xn = ((x - mean.unsqueeze(1)) / std.unsqueeze(1)).clamp(-10.0, 10.0)
        xn = torch.where(missing, zero, xn)

        cells = self.cell_encoder(torch.stack([xn, missing.to(x.dtype)], dim=-1))
        cells = cells + self.feature_emb_proj(
            self._feature_codes(b, n_feat, x.device)
        ).unsqueeze(1)

        # stage 1: column encoder — attention along rows, keys from context rows
        d = self.config.d_model
        key_col = context_mask.repeat_interleave(n_feat, dim=0)  # (B*F, R)
        h = cells.permute(0, 2, 1, 3).reshape(b * n_feat, r, d)
        for block in self.col_blocks:
            h = block(h, key_mask=key_col)
        cells = h.view(b, n_feat, r, d).permute(0, 2, 1, 3)

        # stage 2: row compression — a CLS token attends over each row's F cells
        rows = cells.reshape(b * r, n_feat, d)
        cls = self.row_cls.view(1, 1, d).expand(b * r, 1, d)
        row_vec = self.compress(cls, self.compress_norm(rows)).view(b, r, d)

        # inject the target into context rows; query rows get the mask embedding
        y_tok = torch.where(ctx, self.y_encoder(y_norm.unsqueeze(-1)), self.y_mask_emb)
        row_vec = row_vec + y_tok

        # stage 3: ICL over rows — query rows attend to context rows (the metric)
        for block in self.icl_blocks:
            row_vec = block(row_vec, key_mask=context_mask)

        return self.head(self.head_norm(row_vec))

    def _feature_codes(
        self, batch_size: int, n_feat: int, device: torch.device
    ) -> torch.Tensor:
        shape = (batch_size, n_feat, self.config.feature_emb_dim)
        if self.training:
            return torch.randn(shape, device=device)
        key = (batch_size, n_feat, str(device))
        codes = self._eval_codes.get(key)
        if codes is None:
            gen = torch.Generator().manual_seed(0)
            codes = torch.randn(shape, generator=gen).to(device)
            self._eval_codes[key] = codes
        return codes
