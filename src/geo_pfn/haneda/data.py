"""Haneda Su dataset: loading, feature sets, borehole folds, binning, imputation.

Every helper here is deliberately leakage-safe: fold splits group by borehole,
bin edges and imputer statistics come from the training fold only, and the
soil vocabulary is a fixed constant (uses only the X marginal, never labels).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

TARGET = "Su"
GROUP = "BorSeq"
SOIL_COLUMN = "soil_B02"
SOIL_CODE = "soil_code"

# Soil types with n >= 30 in the pilot CSV (97.1% of rows), most frequent first;
# everything else maps to the trailing "other" bucket.
SOIL_VOCAB = ("CL", "M", "CH", "MS", "MH", "MC", "S", "ML", "SM")

LOCATION_COLUMNS = ("depth_m", "X", "Y")
CHEAP_COLUMNS = ("Wn", "Gs", "LL", "PL", "rho_t", "e")
GRAIN_COLUMNS = ("gravel_pct", "sand_pct", "silt_pct", "clay_pct")


class FeatureSet(str, Enum):
    """Feature-layer ablation arms (docs/haneda-experiment-plan.md §2)."""

    L = "L"  # location only
    LC = "LC"  # + cheap geotech
    LCS = "LCS"  # + soil class
    LCSG = "LCSG"  # + grain-size fractions (full)
    LSG = "LSG"  # location + geology, no cheap geotech


_FEATURE_COLUMNS: dict[FeatureSet, tuple[str, ...]] = {
    FeatureSet.L: LOCATION_COLUMNS,
    FeatureSet.LC: LOCATION_COLUMNS + CHEAP_COLUMNS,
    FeatureSet.LCS: LOCATION_COLUMNS + CHEAP_COLUMNS + (SOIL_CODE,),
    FeatureSet.LCSG: LOCATION_COLUMNS + CHEAP_COLUMNS + (SOIL_CODE,) + GRAIN_COLUMNS,
    FeatureSet.LSG: LOCATION_COLUMNS + (SOIL_CODE,) + GRAIN_COLUMNS,
}


class Imputation(str, Enum):
    """Missing-value strategies (docs/haneda-experiment-plan.md §4)."""

    NATIVE = "native"  # NaN passed straight to the model
    MEAN = "mean"  # train-fold column means, missingness hidden
    KNN = "knn"  # KNNImputer(5) in train-fold-standardized space
    MEAN_INDICATOR = "mean+ind"  # mean fill + 0/1 missing-indicator columns
    PHYSICS_E = "physics-e"  # e rebuilt from Gs/W/rho_t, other NaNs kept


def feature_columns(feature_set: FeatureSet) -> list[str]:
    return list(_FEATURE_COLUMNS[FeatureSet(feature_set)])


def categorical_indices(feature_set: FeatureSet) -> list[int]:
    """Positions of categorical columns (the soil code) within the feature set."""
    cols = feature_columns(feature_set)
    return [cols.index(SOIL_CODE)] if SOIL_CODE in cols else []


def encode_soil(soil: pd.Series) -> np.ndarray:
    """Integer-code soil classes: SOIL_VOCAB order, unknown/rare -> len(vocab)."""
    index = {name: code for code, name in enumerate(SOIL_VOCAB)}
    return soil.map(lambda s: index.get(s, len(SOIL_VOCAB))).to_numpy(dtype=np.float64)


def physics_e(df: pd.DataFrame) -> np.ndarray:
    """Void ratio from phase relations: e = Gs * (1 + W/100) / rho_t - 1 (rho_w = 1).

    Validated on the pilot CSV: median |e - physics_e| = 0.0007 on observed rows.
    """
    return (df["Gs"] * (1.0 + df["W"] / 100.0) / df["rho_t"] - 1.0).to_numpy()


def load_haneda(path: str) -> pd.DataFrame:
    """Load the pilot CSV and attach the encoded soil column."""
    df = pd.read_csv(path, dtype={"soil_B02N": str, SOIL_COLUMN: str})
    required = {
        TARGET,
        GROUP,
        SOIL_COLUMN,
        "W",
        *LOCATION_COLUMNS,
        *CHEAP_COLUMNS,
        *GRAIN_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing expected columns: {sorted(missing)}")
    if df[TARGET].isna().any() or (df[TARGET] <= 0).any():
        raise ValueError("Su must be present and positive on every row")
    df = df.copy()
    df[SOIL_CODE] = encode_soil(df[SOIL_COLUMN])
    return df


def borehole_folds(
    groups: np.ndarray, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold over boreholes; every row appears in exactly one test fold."""
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(groups)), groups=groups))


def quantile_bin_labels(
    y_train: np.ndarray, y: np.ndarray, n_bins: int = 4
) -> np.ndarray:
    """Class labels from train-fold quantile edges (edges never see ``y``)."""
    edges = np.quantile(y_train, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.searchsorted(edges, y, side="right")


def prepare_fold(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    imputation: Imputation,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrices for one fold with the imputation strategy applied.

    All statistics (means, KNN neighbours, scaler, indicator support) are fitted
    on the training fold only. PHYSICS_E is row-local, so it needs no fitting.
    """
    cols = feature_columns(feature_set)
    x = df[cols].to_numpy(dtype=np.float64)
    imputation = Imputation(imputation)

    if imputation is Imputation.PHYSICS_E and "e" in cols:
        e_col = cols.index("e")
        x[:, e_col] = np.where(np.isnan(x[:, e_col]), physics_e(df), x[:, e_col])

    x_train, x_test = x[train_idx], x[test_idx]
    if imputation in (Imputation.NATIVE, Imputation.PHYSICS_E):
        return x_train, x_test

    if imputation is Imputation.KNN:
        scaler = StandardScaler().fit(x_train)
        imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
        z_train = imputer.fit_transform(scaler.transform(x_train))
        z_test = imputer.transform(scaler.transform(x_test))
        return scaler.inverse_transform(z_train), scaler.inverse_transform(z_test)

    add_indicator = imputation is Imputation.MEAN_INDICATOR
    imputer = SimpleImputer(
        strategy="mean", keep_empty_features=True, add_indicator=add_indicator
    )
    return imputer.fit_transform(x_train), imputer.transform(x_test)
