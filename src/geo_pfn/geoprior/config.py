"""Config for the geo-SCM prior (docs/geo-scm-design.md §6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class GeoPriorConfig:
    """Distribution over synthetic borehole tables.

    Each sample is one borehole: rows ordered by depth, segmented into soil
    layers with piecewise-smooth latent evolution (gentle within a layer, a
    jump across a boundary), mapped through a random MLP-SCM to observed cheap
    features and a downstream target. Defaults are calibrated to the Haneda
    pilot CSV (see design doc §1).
    """

    # table shape
    min_rows: int = 30
    max_rows: int = 300
    min_features: int = 8
    max_features: int = 28  # ~19 x 1.5, headroom for extension

    # stratigraphy
    min_layers: int = 1
    max_layers: int = 9
    layer_lambda: float = 2.2  # K = 1 + Poisson(layer_lambda), clipped
    spacing_mean: float = 1.5  # metres between specimens
    surface_depth: float = 1.5  # shallowest specimen depth (m)

    # piecewise-smooth latent field
    latent_dim: int = 4
    within_step: float = 0.5  # per-sqrt-metre random-walk std within a layer
    jump_ratio_min: float = 1.3  # boundary jump / adjacent within-layer step
    jump_ratio_max: float = 2.0
    depth_trend_scale: float = (
        2.2  # strength of the monotone depth driver on the latent
    )

    # SCM mapping latent -> observations
    scm_min_depth: int = 1
    scm_max_depth: int = 3
    scm_min_width: int = 8
    scm_max_width: int = 24
    obs_noise: float = 0.08

    # observation layer
    n_soil_types: int = 10
    block_missing_prob: float = 0.3  # prob a droppable column is absent per hole
    min_train_frac: float = 0.3
    max_train_frac: float = 0.8

    # site assembly (multi-borehole tables; docs/geo-scm-design.md §8)
    p_single: float = 0.3  # fraction of tables that are a single borehole
    max_holes: int = 8  # boreholes per multi-hole site
    min_hole_rows: int = 8
    site_field_scale: float = 1.5  # strength of the smooth spatial field on the latent
    min_anchors: int = 1  # sparse-target holes keep this many depth-spread anchors
    max_anchors: int = 6

    def __post_init__(self) -> None:
        if not 8 <= self.min_rows <= self.max_rows:
            raise ValueError("need 8 <= min_rows <= max_rows")
        if not 2 <= self.min_features <= self.max_features:
            raise ValueError("need 2 <= min_features <= max_features")
        if not 1 <= self.min_layers <= self.max_layers:
            raise ValueError("need 1 <= min_layers <= max_layers")
        if self.layer_lambda < 0:
            raise ValueError("layer_lambda must be >= 0")
        if self.spacing_mean <= 0:
            raise ValueError("spacing_mean must be positive")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.within_step <= 0:
            raise ValueError("within_step must be positive")
        if not 1.0 <= self.jump_ratio_min <= self.jump_ratio_max:
            raise ValueError("need 1 <= jump_ratio_min <= jump_ratio_max")
        if self.depth_trend_scale < 0:
            raise ValueError("depth_trend_scale must be >= 0")
        if not 1 <= self.scm_min_depth <= self.scm_max_depth:
            raise ValueError("need 1 <= scm_min_depth <= scm_max_depth")
        if not 1 <= self.scm_min_width <= self.scm_max_width:
            raise ValueError("need 1 <= scm_min_width <= scm_max_width")
        if not 2 <= self.n_soil_types:
            raise ValueError("n_soil_types must be >= 2")
        if not 0.0 <= self.block_missing_prob < 1.0:
            raise ValueError("block_missing_prob must be in [0, 1)")
        if not 0.0 < self.min_train_frac <= self.max_train_frac < 1.0:
            raise ValueError("train fractions must satisfy 0 < min <= max < 1")
        if not 0.0 <= self.p_single <= 1.0:
            raise ValueError("p_single must be in [0, 1]")
        if self.max_holes < 2:
            raise ValueError("max_holes must be >= 2")
        if self.min_hole_rows < 2:
            raise ValueError("min_hole_rows must be >= 2")
        if self.site_field_scale < 0:
            raise ValueError("site_field_scale must be >= 0")
        if not 1 <= self.min_anchors <= self.max_anchors:
            raise ValueError("need 1 <= min_anchors <= max_anchors")
