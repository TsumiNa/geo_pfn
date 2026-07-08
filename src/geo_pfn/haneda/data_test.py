"""Tests for geo_pfn.haneda.data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_pfn.haneda.data import (
    SOIL_VOCAB,
    FeatureSet,
    Imputation,
    borehole_folds,
    categorical_indices,
    encode_soil,
    feature_columns,
    load_haneda,
    physics_e,
    prepare_fold,
    quantile_bin_labels,
)


def make_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gs = rng.dirichlet(np.ones(4), size=n) * 100.0
    df = pd.DataFrame(
        {
            "BorSeq": rng.integers(0, 10, size=n),
            "depth_m": -rng.uniform(1, 60, size=n),
            "X": rng.uniform(-50_000, -49_000, size=n),
            "Y": rng.uniform(-5_000, -4_000, size=n),
            "soil_B02N": "0500",
            "soil_B02": rng.choice(["CL", "M", "CH", "ZZZ"], size=n),
            "Wn": rng.uniform(30, 120, size=n),
            "Gs": rng.uniform(2.6, 2.8, size=n),
            "LL": rng.uniform(50, 120, size=n),
            "PL": rng.uniform(20, 50, size=n),
            "gravel_pct": gs[:, 0],
            "sand_pct": gs[:, 1],
            "silt_pct": gs[:, 2],
            "clay_pct": gs[:, 3],
            "rho_t": rng.uniform(1.3, 1.9, size=n),
            "W": rng.uniform(30, 120, size=n),
            "e": rng.uniform(1.0, 3.0, size=n),
            "Su": rng.uniform(1, 150, size=n),
        }
    )
    df["qu"] = 2 * df["Su"]
    df["soil_code"] = encode_soil(df["soil_B02"])
    # block missingness like the real data: e and grain size NaN'd per borehole
    e_missing = df["BorSeq"] < 3
    df.loc[e_missing, "e"] = np.nan
    gs_missing = df["BorSeq"] % 2 == 0
    df.loc[gs_missing, ["gravel_pct", "sand_pct", "silt_pct", "clay_pct"]] = np.nan
    return df


def test_encode_soil_vocab_and_other() -> None:
    codes = encode_soil(pd.Series(["CL", "SM", "SPC", "NO"]))
    assert codes[0] == 0.0  # most frequent class first
    assert codes[1] == len(SOIL_VOCAB) - 1
    assert codes[2] == codes[3] == float(len(SOIL_VOCAB))  # unknown -> other


def test_feature_columns_and_categoricals() -> None:
    assert feature_columns(FeatureSet.L) == ["depth_m", "X", "Y"]
    assert len(feature_columns(FeatureSet.LCS)) == 10
    assert len(feature_columns(FeatureSet.LCSG)) == 14
    assert categorical_indices(FeatureSet.L) == []
    lcs = feature_columns(FeatureSet.LCS)
    assert lcs[categorical_indices(FeatureSet.LCS)[0]] == "soil_code"


def test_borehole_folds_partition_and_group_integrity() -> None:
    groups = np.repeat(np.arange(12), 5)
    folds = borehole_folds(groups, n_splits=4, seed=0)
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test) == list(range(len(groups)))
    for train_idx, test_idx in folds:
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_quantile_bins_balanced_and_train_only() -> None:
    y_train = np.arange(100, dtype=np.float64)
    labels = quantile_bin_labels(y_train, y_train, n_bins=4)
    assert (np.bincount(labels, minlength=4) == 25).all()
    # test values far outside the train range clip into the edge bins
    y_test = np.array([-1e6, 1e6])
    assert quantile_bin_labels(y_train, y_test, n_bins=4).tolist() == [0, 3]


def test_prepare_fold_native_keeps_nan() -> None:
    df = make_df()
    train_idx, test_idx = np.arange(0, 60), np.arange(60, 80)
    x_train, x_test = prepare_fold(
        df, FeatureSet.LCSG, Imputation.NATIVE, train_idx, test_idx
    )
    assert x_train.shape[1] == 14
    assert np.isnan(x_train).any() and np.isnan(x_test).any()


def test_prepare_fold_mean_uses_train_statistics() -> None:
    df = make_df()
    cols = feature_columns(FeatureSet.LC)
    e_col = cols.index("e")
    train_idx, test_idx = np.arange(0, 60), np.arange(60, 80)
    x_train, x_test = prepare_fold(
        df, FeatureSet.LC, Imputation.MEAN, train_idx, test_idx
    )
    assert not np.isnan(x_train).any() and not np.isnan(x_test).any()
    raw = df[cols].to_numpy()
    train_mean = np.nanmean(raw[train_idx, e_col])
    was_nan = np.isnan(raw[test_idx, e_col])
    assert was_nan.any()
    np.testing.assert_allclose(x_test[was_nan, e_col], train_mean)


def test_prepare_fold_indicator_appends_columns() -> None:
    df = make_df()
    train_idx, test_idx = np.arange(0, 60), np.arange(60, 80)
    x_train, x_test = prepare_fold(
        df, FeatureSet.LCSG, Imputation.MEAN_INDICATOR, train_idx, test_idx
    )
    assert x_train.shape[1] == x_test.shape[1] == 14 + 5  # e + 4 grain-size flags
    assert set(np.unique(x_train[:, 14:])) <= {0.0, 1.0}


def test_prepare_fold_knn_fills_all() -> None:
    df = make_df()
    train_idx, test_idx = np.arange(0, 60), np.arange(60, 80)
    x_train, x_test = prepare_fold(
        df, FeatureSet.LCSG, Imputation.KNN, train_idx, test_idx
    )
    assert not np.isnan(x_train).any() and not np.isnan(x_test).any()
    # observed cells come back on the original scale, untouched
    raw = df[feature_columns(FeatureSet.LCSG)].to_numpy()[train_idx]
    observed = ~np.isnan(raw)
    np.testing.assert_allclose(x_train[observed], raw[observed], atol=1e-9)


def test_prepare_fold_physics_e_reconstructs_only_e() -> None:
    df = make_df()
    cols = feature_columns(FeatureSet.LCSG)
    e_col = cols.index("e")
    train_idx, test_idx = np.arange(0, 60), np.arange(60, 80)
    x_train, x_test = prepare_fold(
        df, FeatureSet.LCSG, Imputation.PHYSICS_E, train_idx, test_idx
    )
    x = np.concatenate([x_train, x_test])
    idx = np.concatenate([train_idx, test_idx])
    raw_e = df["e"].to_numpy()[idx]
    expected = physics_e(df)
    np.testing.assert_allclose(
        x[np.isnan(raw_e), e_col], expected[idx][np.isnan(raw_e)]
    )
    np.testing.assert_allclose(x[~np.isnan(raw_e), e_col], raw_e[~np.isnan(raw_e)])
    grain = [cols.index(c) for c in ("gravel_pct", "sand_pct", "silt_pct", "clay_pct")]
    assert np.isnan(x[:, grain]).any()  # grain-size NaNs are preserved


def test_load_haneda_validates(tmp_path) -> None:
    df = make_df()
    good = tmp_path / "good.csv"
    df.drop(columns=["soil_code"]).to_csv(good, index=False)
    loaded = load_haneda(str(good))
    assert "soil_code" in loaded.columns

    df_bad = df.drop(columns=["soil_code", "rho_t"])
    bad = tmp_path / "bad.csv"
    df_bad.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_haneda(str(bad))

    df_neg = df.drop(columns=["soil_code"]).copy()
    df_neg.loc[0, "Su"] = -1.0
    neg = tmp_path / "neg.csv"
    df_neg.to_csv(neg, index=False)
    with pytest.raises(ValueError, match="positive"):
        load_haneda(str(neg))
