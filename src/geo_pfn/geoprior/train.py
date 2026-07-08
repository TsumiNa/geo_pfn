"""Pretrain a PFN on the geo-SCM prior: ``python -m geo_pfn.geoprior.train``.

Milestone 1 — get the geo prior training end-to-end on the *existing* MiniPFN
architecture, unchanged. The continuous borehole target is quantile-binned into
classes (edges from the context rows only, so no leak) and the model is trained
with the standard PFN cross-entropy on the query rows. A random per-row context
split gives every borehole a spread of same-hole context rows plus other-hole
rows, so a multi-hole table trains exactly the anchor + neighbour transfer the
model must eventually do. (The site sampler's designed context mask — dense
neighbours + sparse targets — needs the model to accept a per-row mask; that
lands with the two-stage-attention upgrade, docs/geo-scm-design.md §9.)
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
from torch.nn import functional as func

from geo_pfn.geoprior.config import GeoPriorConfig
from geo_pfn.geoprior.site import sample_geo_site_batch
from geo_pfn.minipfn.config import ModelConfig
from geo_pfn.minipfn.model import MiniPFN, mask_class_logits
from geo_pfn.minipfn.train import resolve_device, save_checkpoint


@dataclass(kw_only=True)
class GeoTrainConfig:
    """Pretraining hyperparameters for the geo-PFN milestone-1 run."""

    steps: int = 12_000
    batch_size: int = 16
    n_bins: int = 4
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    log_every: int = 100
    checkpoint_every: int = 2_000
    out_path: str = "checkpoints/geopfn.pt"
    min_train_frac: float = 0.3
    max_train_frac: float = 0.8

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if not 2 <= self.n_bins <= 4:
            raise ValueError("n_bins must be in [2, 4] (MiniPFN head is 4-way)")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be in [0, steps]")
        if not 0.0 < self.min_train_frac <= self.max_train_frac < 1.0:
            raise ValueError("train fractions must satisfy 0 < min <= max < 1")


def geo_task(
    prior_cfg: GeoPriorConfig,
    batch_size: int,
    n_bins: int,
    train_frac: tuple[float, float],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sample a geo batch, random-split rows, and quantile-bin the target.

    Returns (x (B, R, F), y_binned (B, R) long, train_size). Bin edges come from
    the first ``train_size`` (context) rows of each table, so labels never leak.
    """
    batch = sample_geo_site_batch(prior_cfg, batch_size, generator)
    b, r, _ = batch.x.shape
    perm = torch.argsort(torch.rand(b, r, generator=generator), dim=1)
    x = torch.gather(batch.x, 1, perm.unsqueeze(-1).expand_as(batch.x))
    y = torch.gather(batch.y, 1, perm)

    frac = train_frac[0] + (train_frac[1] - train_frac[0]) * float(
        torch.rand(1, generator=generator).item()
    )
    train_size = min(max(int(r * frac), n_bins), r - 1)

    ctx = y[:, :train_size]
    qs = torch.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = torch.quantile(ctx, qs, dim=1).T  # (B, n_bins - 1)
    labels = (y.unsqueeze(-1) > edges.unsqueeze(1)).sum(-1)
    return x, labels.long(), train_size


def train(
    train_cfg: GeoTrainConfig, model_cfg: ModelConfig, prior_cfg: GeoPriorConfig
) -> tuple[MiniPFN, list[dict[str, float]]]:
    device = resolve_device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)
    sampler = torch.Generator().manual_seed(train_cfg.seed)

    model = MiniPFN(model_cfg).to(device)
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
    n_classes = torch.full((train_cfg.batch_size,), train_cfg.n_bins, device=device)

    history: list[dict[str, float]] = []
    win_loss = win_acc = 0.0
    win_n = 0
    t_start = time.time()
    model.train()
    for step in range(train_cfg.steps):
        x, y, train_size = geo_task(
            prior_cfg,
            train_cfg.batch_size,
            train_cfg.n_bins,
            (train_cfg.min_train_frac, train_cfg.max_train_frac),
            sampler,
        )
        x, y = x.to(device), y.to(device)
        logits = mask_class_logits(model(x, y, train_size), n_classes)
        y_test = y[:, train_size:]
        loss = func.cross_entropy(logits.flatten(0, 1), y_test.flatten())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc = (logits.argmax(-1) == y_test).float().mean()
        win_loss += float(loss.detach())
        win_acc += float(acc)
        win_n += 1
        if (step + 1) % train_cfg.log_every == 0:
            entry = {
                "step": step + 1,
                "loss": win_loss / win_n,
                "acc": win_acc / win_n,
                "lr": float(scheduler.get_last_lr()[0]),
            }
            history.append(entry)
            speed = (step + 1) / (time.time() - t_start)
            print(
                f"step {entry['step']:>6d}  loss {entry['loss']:.4f}  "
                f"acc {entry['acc']:.4f}  lr {entry['lr']:.2e}  {speed:.1f} it/s"
            )
            win_loss = win_acc = 0.0
            win_n = 0
        if (step + 1) % train_cfg.checkpoint_every == 0:
            save_checkpoint(
                train_cfg.out_path, model, prior_cfg, train_cfg, step + 1, history
            )

    save_checkpoint(
        train_cfg.out_path, model, prior_cfg, train_cfg, train_cfg.steps, history
    )
    print(f"saved {train_cfg.out_path}")
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="checkpoints/geopfn.pt")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    args = parser.parse_args()

    train_cfg = GeoTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
        warmup_steps=min(500, max(1, args.steps // 10)),
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        out_path=args.out,
        log_every=args.log_every,
    )
    model_cfg = ModelConfig(
        d_model=args.d_model, n_layers=args.n_layers, max_classes=args.n_bins
    )
    train(train_cfg, model_cfg, GeoPriorConfig())


if __name__ == "__main__":
    main()
