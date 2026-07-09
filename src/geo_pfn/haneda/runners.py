"""Classical baselines and metrics for the Haneda real-data evaluation."""

from __future__ import annotations

import numpy as np
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
