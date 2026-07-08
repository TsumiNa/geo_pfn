"""Model runners and metrics for the Haneda real-data evaluation.

The mini-PFN was pretrained on 60-160-row tables, so it never sees the ~2800
training rows of a fold at once: predictions ensemble E forward passes, each
over a class-stratified random context of ``ctx_size`` rows, averaging the
softmax probabilities (mirroring TabPFN-style ensembling). Test rows attend
only to context rows inside the model, so chunking test rows is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geo_pfn.minipfn.model import MiniPFN, mask_class_logits


@dataclass(kw_only=True)
class ContextConfig:
    """How the mini-PFN consumes a training fold that exceeds its context size."""

    ctx_size: int = 128
    n_ensembles: int = 16
    test_chunk: int = 512

    def __post_init__(self) -> None:
        if self.ctx_size < 2:
            raise ValueError("ctx_size must be >= 2")
        if self.n_ensembles < 1:
            raise ValueError("n_ensembles must be >= 1")
        if self.test_chunk < 1:
            raise ValueError("test_chunk must be >= 1")


def _stratified_context(
    y: torch.Tensor, ctx_size: int, generator: torch.Generator
) -> torch.Tensor:
    """Random context indices with every present class represented.

    Quotas are proportional to class frequency (>= 1 each); the remainder is
    filled from the not-yet-picked rows uniformly at random.
    """
    n = len(y)
    if n <= ctx_size:
        return torch.randperm(n, generator=generator)
    picked = []
    for cls in y.unique():
        idx = (y == cls).nonzero().squeeze(1)
        quota = max(1, round(ctx_size * len(idx) / n))
        perm = torch.randperm(len(idx), generator=generator)[:quota]
        picked.append(idx[perm])
    picked_t = torch.cat(picked)
    if len(picked_t) > ctx_size:
        keep = torch.randperm(len(picked_t), generator=generator)[:ctx_size]
        return picked_t[keep]
    if len(picked_t) < ctx_size:
        mask = torch.ones(n, dtype=torch.bool)
        mask[picked_t] = False
        rest = mask.nonzero().squeeze(1)
        extra = torch.randperm(len(rest), generator=generator)
        picked_t = torch.cat([picked_t, rest[extra[: ctx_size - len(picked_t)]]])
    return picked_t


def minipfn_predict_proba(
    model: MiniPFN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    config: ContextConfig,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """(n_test, n_classes) probabilities, ensemble-averaged over random contexts."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    ctx_x_all = torch.tensor(x_train, dtype=torch.float32)
    ctx_y_all = torch.tensor(y_train, dtype=torch.long)
    test_x = torch.tensor(x_test, dtype=torch.float32)
    n_classes_t = torch.tensor([n_classes], device=device)

    total = torch.zeros(len(test_x), n_classes)
    for _ in range(config.n_ensembles):
        idx = _stratified_context(ctx_y_all, config.ctx_size, generator)
        ctx_x, ctx_y = ctx_x_all[idx], ctx_y_all[idx]
        for start in range(0, len(test_x), config.test_chunk):
            chunk = test_x[start : start + config.test_chunk]
            x = torch.cat([ctx_x, chunk]).unsqueeze(0).to(device)
            y = torch.cat([ctx_y, torch.zeros(len(chunk), dtype=torch.long)])
            with torch.no_grad():
                logits = model(x, y.unsqueeze(0).to(device), train_size=len(idx))
                logits = mask_class_logits(logits, n_classes_t)
                probs = torch.softmax(logits[0, :, :n_classes], dim=-1)
            total[start : start + len(chunk)] += probs.cpu()
    return (total / config.n_ensembles).numpy()


def make_tabpfn_v2(task: str, device: str, n_estimators: int | None = None):
    """TabPFN v2 estimator (lazy import; weights download on first fit)."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    est_cls = {"regression": TabPFNRegressor, "classification": TabPFNClassifier}[task]
    overrides: dict = {"device": device}
    if n_estimators is not None:
        overrides["n_estimators"] = n_estimators
    return est_cls.create_default_for_version(ModelVersion.V2, **overrides)


def make_baseline(name: str, task: str):
    """Classical baselines; ``linear`` and ``depth`` mean-impute internally."""
    if name == "hgbt":
        if task == "regression":
            return HistGradientBoostingRegressor(random_state=0)
        return HistGradientBoostingClassifier(random_state=0)
    if name in ("linear", "depth"):
        head = (
            Ridge(alpha=1.0)
            if task == "regression"
            else LogisticRegression(max_iter=1000)
        )
        return make_pipeline(
            SimpleImputer(strategy="mean", keep_empty_features=True),
            StandardScaler(),
            head,
        )
    if name == "dummy":
        if task == "regression":
            return DummyRegressor(strategy="mean")
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"unknown baseline: {name}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": 1.0 - float((err**2).sum()) / ss_tot if ss_tot > 0 else 0.0,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
