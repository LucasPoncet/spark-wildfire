from dataclasses import replace

import numpy as np
import pytest

from src.config.multi_source_localization_configuration import (
    MultiSourceLocalizationConfiguration,
)
from src.config.simulation_configuration import LocalizationConfiguration
from src.spark.acoustic.free_field_propagation import (
    render_multi_source_receiver_signals,
)
from src.spark.acoustic.receiver_layout import build_ring_receiver_positions_xyz
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.utils.array_types import Float64Array

SCENE_DOMAIN_EXTENT_M: float = 40.0
SCENE_RING_RADIUS_M: float = 18.0
SCENE_RECEIVER_HEIGHT_M: float = 1.5
SCENE_SOURCE_HEIGHT_M: float = 0.5
SCENE_CLIP_DURATION_S: float = 2.0
SCENE_REFERENCE_DISTANCE_M: float = 1.0


@pytest.fixture(scope="session")
def multi_source(
    localization: LocalizationConfiguration,
) -> MultiSourceLocalizationConfiguration:
    """The real configuration, made cheap enough to run inside a test."""
    return replace(
        localization.multi_source,
        correlation=replace(
            localization.multi_source.correlation, window_duration_s=0.25
        ),
        steered_response_power=replace(
            localization.multi_source.steered_response_power,
            coarse_grid_spacing_m=2.0,
            refinement_levels=2,
            candidate_height_m=SCENE_SOURCE_HEIGHT_M,
        ),
        clustering=replace(
            localization.multi_source.clustering,
            window_duration_s=0.5,
            clustering_radius_m=6.0,
            minimum_cluster_size=2,
        ),
    )


@pytest.fixture(scope="session")
def scene_receivers_xyz_m() -> Float64Array:
    return build_ring_receiver_positions_xyz(
        0.5 * SCENE_DOMAIN_EXTENT_M,
        0.5 * SCENE_DOMAIN_EXTENT_M,
        SCENE_RING_RADIUS_M,
        4,
        0.0,
        SCENE_RECEIVER_HEIGHT_M,
    )


def render_scene_signals(
    source_positions_xy_m: Float64Array,
    source_amplitude_scales: Float64Array,
    receivers_xyz_m: Float64Array,
    conditions: AtmosphericConditions,
    sample_rate_hz: int,
    seed: int = 0,
) -> Float64Array:
    """Renders independent noise sources through the real forward channel."""
    generator = np.random.default_rng(seed)
    sample_count = round(SCENE_CLIP_DURATION_S * sample_rate_hz)
    positions_xyz_m = np.column_stack(
        (
            np.atleast_2d(source_positions_xy_m),
            np.full(
                np.atleast_2d(source_positions_xy_m).shape[0], SCENE_SOURCE_HEIGHT_M
            ),
        )
    )
    return render_multi_source_receiver_signals(
        [
            generator.standard_normal(sample_count)
            for _ in range(positions_xyz_m.shape[0])
        ],
        positions_xyz_m,
        np.asarray(source_amplitude_scales, dtype=np.float64),
        receivers_xyz_m,
        sample_rate_hz,
        conditions,
        SCENE_REFERENCE_DISTANCE_M,
    )
