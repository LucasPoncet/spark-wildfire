import numpy as np
import pytest

from src.config.multi_source_localization_configuration import (
    MultiSourceLocalizationConfiguration,
)
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.spark.inverse.windowed_position_clustering import (
    cluster_positions_by_radius,
    estimate_positions_by_windowed_clustering,
)
from src.utils.array_types import Float64Array
from tests.spark.inverse.conftest import (
    SCENE_DOMAIN_EXTENT_M,
    render_scene_signals,
)

RECOVERY_TOLERANCE_M: float = 3.0


def test_two_separated_point_clouds_form_two_clusters() -> None:
    generator = np.random.default_rng(0)
    positions_xy_m = np.vstack(
        (
            np.array([10.0, 10.0]) + 0.5 * generator.standard_normal((20, 2)),
            np.array([30.0, 30.0]) + 0.5 * generator.standard_normal((15, 2)),
        )
    )
    clusters = cluster_positions_by_radius(positions_xy_m, 5.0, 3)
    assert len(clusters) == 2
    assert clusters[0].window_count == 20
    assert clusters[1].window_count == 15
    assert np.allclose(clusters[0].centroid_xy_m, [10.0, 10.0], atol=0.5)


def test_a_thin_cluster_is_dropped_below_the_minimum_membership() -> None:
    positions_xy_m = np.vstack(
        (np.tile([10.0, 10.0], (10, 1)), np.tile([30.0, 30.0], (2, 1)))
    )
    clusters = cluster_positions_by_radius(positions_xy_m, 5.0, 3)
    assert len(clusters) == 1
    assert np.allclose(clusters[0].centroid_xy_m, [10.0, 10.0])


def test_clusters_come_back_most_populous_first() -> None:
    positions_xy_m = np.vstack(
        (np.tile([10.0, 10.0], (4, 1)), np.tile([30.0, 30.0], (9, 1)))
    )
    clusters = cluster_positions_by_radius(positions_xy_m, 5.0, 3)
    assert [cluster.window_count for cluster in clusters] == [9, 4]


def test_the_cluster_spread_reports_the_scatter_of_its_members() -> None:
    positions_xy_m = np.array([[10.0, 10.0], [12.0, 10.0], [8.0, 10.0]])
    cluster = cluster_positions_by_radius(positions_xy_m, 5.0, 3)[0]
    assert cluster.spread_m == pytest.approx(np.sqrt(8.0 / 3.0))


def test_a_non_positive_clustering_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="clustering_radius_m must be positive"):
        cluster_positions_by_radius(np.zeros((3, 2)), 0.0, 1)


def test_one_source_is_recovered_as_one_cluster(
    scene_receivers_xyz_m: Float64Array,
    atmosphere: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
) -> None:
    true_position_xy_m = np.array([12.0, 27.0])
    clustering = estimate_positions_by_windowed_clustering(
        render_scene_signals(
            np.array([true_position_xy_m]),
            np.array([1.0]),
            scene_receivers_xyz_m,
            atmosphere,
            sample_rate_hz,
            seed=5,
        ),
        scene_receivers_xyz_m,
        sample_rate_hz,
        SCENE_DOMAIN_EXTENT_M,
        SCENE_DOMAIN_EXTENT_M,
        atmosphere,
        multi_source,
    )
    assert clustering.estimated_source_count == 1
    assert clustering.window_maxima_xy_m.shape[0] > 1
    assert (
        float(np.linalg.norm(clustering.clusters[0].centroid_xy_m - true_position_xy_m))
        < RECOVERY_TOLERANCE_M
    )


def test_channels_shorter_than_one_window_are_rejected(
    scene_receivers_xyz_m: Float64Array,
    atmosphere: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
) -> None:
    with pytest.raises(ValueError, match="shorter than one clustering window"):
        estimate_positions_by_windowed_clustering(
            np.zeros((4, 100)),
            scene_receivers_xyz_m,
            sample_rate_hz,
            SCENE_DOMAIN_EXTENT_M,
            SCENE_DOMAIN_EXTENT_M,
            atmosphere,
            multi_source,
        )
