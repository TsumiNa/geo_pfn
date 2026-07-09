"""Pretrain the two-stage geo-PFN: ``python -m geo_pfn.geopfn.train``.

Trains on the multi-borehole site prior using its designed context mask (dense
neighbour holes fully observed, sparse target holes with a few depth-spread
anchors), so the model learns exactly the sparse-anchor cross-borehole transfer
the hypothesis is about. The continuous target is context-normalized and fit
with the bar-distribution head (regression + uncertainty).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch

from geo_pfn.geopfn.model import GeoPFN, GeoPFNConfig
from geo_pfn.geoprior.config import GeoPriorConfig
from geo_pfn.geoprior.site import sample_geo_site_batch
from geo_pfn.util import resolve_device, save_checkpoint


@dataclass(kw_only=True)
class GeoPFNTrainConfig:
    steps: int = 12_000
    batch_size: int = 16
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    log_every: int = 100
    checkpoint_every: int = 2_000
    out_path: str = "checkpoints/geopfn2stage.pt"

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be in [0, steps]")


def normalize_target(
    y: torch.Tensor, context_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Context-normalize the target per table. Returns (z, mean, std).

    The std is floored at a fraction of the whole-table spread so a degenerate
    (low-variance) context can't blow z up, and z is clipped to the range the
    bar-distribution can represent; both keep training stable on homogeneous
    boreholes without distorting normal or real (large-scale) targets.
    """
    ctx = context_mask.float()
    count = ctx.sum(1).clamp(min=1)
    mean = (y * ctx).sum(1) / count
    var = (((y - mean.unsqueeze(1)) ** 2) * ctx).sum(1) / count
    floor = 0.1 * y.std(dim=1).clamp(min=1e-6)
    std = torch.maximum(var.sqrt(), floor)
    z = ((y - mean.unsqueeze(1)) / std.unsqueeze(1)).clamp(-8.0, 8.0)
    return z, mean, std


def _step_loss(
    model: GeoPFN,
    prior_cfg: GeoPriorConfig,
    batch_size: int,
    g: torch.Generator,
    device,
) -> tuple[torch.Tensor, float]:
    batch = sample_geo_site_batch(prior_cfg, batch_size, g)
    x = batch.x.to(device)
    ctx = batch.context_mask.to(device)
    z, _, _ = normalize_target(batch.y.to(device), ctx)
    logits = model(x, z, ctx)
    query = ~ctx
    logits_q, z_q = logits[query], z[query]
    loss = model.bar.loss(logits_q, z_q)
    with torch.no_grad():
        pred, _ = model.bar.mean_std(logits_q)
        rmse_norm = ((pred - z_q) ** 2).mean().sqrt().item()
    return loss, rmse_norm


def train(
    train_cfg: GeoPFNTrainConfig, model_cfg: GeoPFNConfig, prior_cfg: GeoPriorConfig
) -> tuple[GeoPFN, list[dict[str, float]]]:
    device = resolve_device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)
    sampler = torch.Generator().manual_seed(train_cfg.seed)

    model = GeoPFN(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device.type} params={n_params / 1e6:.2f}M steps={train_cfg.steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < train_cfg.warmup_steps:
            return (step + 1) / max(train_cfg.warmup_steps, 1)
        progress = (step - train_cfg.warmup_steps) / max(
            train_cfg.steps - train_cfg.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history: list[dict[str, float]] = []
    win_loss = win_rmse = 0.0
    win_n = 0
    t_start = time.time()
    model.train()
    for step in range(train_cfg.steps):
        loss, rmse_norm = _step_loss(
            model, prior_cfg, train_cfg.batch_size, sampler, device
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        win_loss += float(loss.detach())
        win_rmse += rmse_norm
        win_n += 1
        if (step + 1) % train_cfg.log_every == 0:
            entry = {
                "step": step + 1,
                "loss": win_loss / win_n,
                "rmse_norm": win_rmse / win_n,
                "lr": float(scheduler.get_last_lr()[0]),
            }
            history.append(entry)
            speed = (step + 1) / (time.time() - t_start)
            print(
                f"step {entry['step']:>6d}  loss {entry['loss']:.4f}  "
                f"rmse_norm {entry['rmse_norm']:.4f}  lr {entry['lr']:.2e}  "
                f"{speed:.1f} it/s",
                flush=True,
            )
            win_loss = win_rmse = 0.0
            win_n = 0
        if (step + 1) % train_cfg.checkpoint_every == 0:
            save_checkpoint(
                train_cfg.out_path, model, prior_cfg, train_cfg, step + 1, history
            )

    save_checkpoint(
        train_cfg.out_path, model, prior_cfg, train_cfg, train_cfg.steps, history
    )
    print(f"saved {train_cfg.out_path}", flush=True)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="checkpoints/geopfn2stage.pt")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--col-layers", type=int, default=2)
    parser.add_argument("--row-layers", type=int, default=4)
    parser.add_argument("--feature-emb-dim", type=int, default=48)
    args = parser.parse_args()

    train_cfg = GeoPFNTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=min(500, max(1, args.steps // 10)),
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        out_path=args.out,
        log_every=args.log_every,
    )
    model_cfg = GeoPFNConfig(
        d_model=args.d_model,
        n_bins=args.n_bins,
        n_heads=args.n_heads,
        col_layers=args.col_layers,
        row_layers=args.row_layers,
        feature_emb_dim=args.feature_emb_dim,
    )
    train(train_cfg, model_cfg, GeoPriorConfig())


if __name__ == "__main__":
    main()
