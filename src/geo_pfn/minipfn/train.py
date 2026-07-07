"""Pretrain the mini-PFN on synthetic tasks: ``python -m geo_pfn.minipfn.train``.

Each step samples a fresh batch of synthetic tasks from the prior, applies the
test-time feature-dropout augmentation, and minimizes cross-entropy on the
test rows — the standard PFN objective. Set ``--drop-task-prob 0`` to train a
vanilla (non missingness-augmented) ablation model.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as func

from geo_pfn.minipfn.config import Augmentation, ModelConfig, PriorConfig, TrainConfig
from geo_pfn.minipfn.model import MiniPFN, mask_class_logits
from geo_pfn.minipfn.prior import (
    PriorBatch,
    random_cell_missing,
    random_test_missing,
    sample_prior_batch,
)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: MiniPFN,
    prior_cfg: PriorConfig,
    train_cfg: TrainConfig,
    step: int,
    history: list[dict[str, float]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "prior_config": asdict(prior_cfg),
            "train_config": asdict(train_cfg),
            "step": step,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[MiniPFN, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MiniPFN(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def _augment(
    batch: PriorBatch, train_cfg: TrainConfig, generator: torch.Generator
) -> torch.Tensor:
    """Apply the configured missingness augmentation to a prior batch."""
    if train_cfg.augmentation is Augmentation.CELLS:
        return random_cell_missing(
            batch.x,
            batch.y,
            batch.n_classes,
            train_cfg.cell_missing_task_prob,
            train_cfg.cell_missing_max_rate,
            train_cfg.informative_missing_prob,
            generator,
        )
    if train_cfg.augmentation is Augmentation.COLUMNS:
        return random_test_missing(
            batch.x,
            batch.train_size,
            train_cfg.drop_feature_task_prob,
            train_cfg.max_drop_frac,
            generator,
        )
    return batch.x


def train(
    train_cfg: TrainConfig, model_cfg: ModelConfig, prior_cfg: PriorConfig
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

    history: list[dict[str, float]] = []
    window_loss, window_acc, window_n = 0.0, 0.0, 0
    t_start = time.time()
    model.train()
    for step in range(train_cfg.steps):
        batch = sample_prior_batch(prior_cfg, train_cfg.batch_size, sampler)
        x = _augment(batch, train_cfg, sampler)
        x = x.to(device)
        y = batch.y.to(device)
        n_classes = batch.n_classes.to(device)

        logits = mask_class_logits(model(x, y, batch.train_size), n_classes)
        y_test = y[:, batch.train_size :]
        loss = func.cross_entropy(logits.flatten(0, 1), y_test.flatten())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc = (logits.argmax(-1) == y_test).float().mean()
        window_loss += float(loss.detach())
        window_acc += float(acc)
        window_n += 1
        if (step + 1) % train_cfg.log_every == 0:
            entry = {
                "step": step + 1,
                "loss": window_loss / window_n,
                "acc": window_acc / window_n,
                "lr": float(scheduler.get_last_lr()[0]),
            }
            history.append(entry)
            speed = (step + 1) / (time.time() - t_start)
            print(
                f"step {entry['step']:>6d}  loss {entry['loss']:.4f}  "
                f"acc {entry['acc']:.4f}  lr {entry['lr']:.2e}  {speed:.1f} it/s"
            )
            window_loss, window_acc, window_n = 0.0, 0.0, 0
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
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="checkpoints/minipfn.pt")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--augmentation",
        type=str,
        default="cells",
        choices=[a.value for a in Augmentation],
        help="missingness augmentation: cells (questionnaire-style), columns, none",
    )
    parser.add_argument(
        "--drop-task-prob",
        type=float,
        default=0.5,
        help="columns mode: probability a task gets test-row feature dropout",
    )
    args = parser.parse_args()

    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=min(500, max(1, args.steps // 10)),
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        out_path=args.out,
        log_every=args.log_every,
        augmentation=Augmentation(args.augmentation),
        drop_feature_task_prob=args.drop_task_prob,
    )
    model_cfg = ModelConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads
    )
    train(train_cfg, model_cfg, PriorConfig())


if __name__ == "__main__":
    main()
