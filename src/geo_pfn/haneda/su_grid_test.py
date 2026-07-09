"""Tests for geo_pfn.haneda.su_grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geo_pfn.haneda.su_grid import build_grid


def test_build_grid_shape_and_extent() -> None:
    df = pd.DataFrame({"X": [0.0, 10.0, 5.0], "Y": [0.0, 20.0, 5.0]})
    depths = np.array([5, 10, 15])
    grid = build_grid(df, nx=4, ny=6, depths=depths)
    assert grid.shape == (3 * 4 * 6, 3)  # depth x nx x ny, cols [depth_m, X, Y]
    assert grid[:, 0].min() == -15 and grid[:, 0].max() == -5  # depth_m negative
    assert grid[:, 1].min() == 0.0 and grid[:, 1].max() == 10.0  # X extent
    assert grid[:, 2].min() == 0.0 and grid[:, 2].max() == 20.0  # Y extent
