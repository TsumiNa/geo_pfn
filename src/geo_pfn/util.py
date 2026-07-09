"""Shared, model-agnostic utilities: device resolution and checkpoint saving."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


def resolve_device(name: str) -> torch.device:
    """Resolve ``"auto"`` to MPS/CUDA/CPU, or pass an explicit device name through."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    prior_cfg: Any,
    train_cfg: Any,
    step: int,
    history: list[dict[str, float]],
) -> None:
    """Save a reload-from-config checkpoint (state dict + dataclass configs).

    The model is stored as ``state_dict`` plus ``model_config = asdict(model.config)``,
    so it reloads via ``Model(Config(**model_config)).load_state_dict(...)`` without
    pickling the model class — refactoring model code never breaks old checkpoints.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config) if is_dataclass(model.config) else {},
            "prior_config": asdict(prior_cfg) if is_dataclass(prior_cfg) else prior_cfg,
            "train_config": asdict(train_cfg) if is_dataclass(train_cfg) else train_cfg,
            "step": step,
            "history": history,
        },
        path,
    )
