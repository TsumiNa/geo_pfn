"""Smoke tests for the pretraining loop and checkpoint round-trip."""

from pathlib import Path

import torch

from geo_pfn.minipfn.config import ModelConfig, PriorConfig, TrainConfig
from geo_pfn.minipfn.train import load_checkpoint, train


def test_short_training_run_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "ckpt.pt"
    train_cfg = TrainConfig(
        steps=12,
        batch_size=4,
        warmup_steps=2,
        device="cpu",
        log_every=6,
        checkpoint_every=1000,
        out_path=str(out),
    )
    model_cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, feature_emb_dim=8)
    prior_cfg = PriorConfig(min_rows=30, max_rows=40, max_features=6)

    model, history = train(train_cfg, model_cfg, prior_cfg)

    assert out.exists()
    assert len(history) == 2
    assert all(0.0 < h["loss"] < 10.0 for h in history)

    loaded, ckpt = load_checkpoint(out, torch.device("cpu"))
    assert ckpt["step"] == 12
    assert ckpt["prior_config"]["max_features"] == 6
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 20, 4, generator=g)
    y = torch.randint(0, 2, (2, 20), generator=g)
    logits = loaded(x, y, train_size=12)
    assert logits.shape == (2, 8, model_cfg.max_classes)
    assert torch.isfinite(logits).all()
