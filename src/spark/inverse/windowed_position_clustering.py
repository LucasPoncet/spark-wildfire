"""A second route to the source count, independent of deflation.

Most multi-source steered-response work leans on speech structure: disjoint
time-frequency support, one source dominant per bin, voice activity detection.
None of that transfers to continuous, stationary, spectrally similar broadband
noise, which is what several fires sound like at once.

The one speech-like property fire does have is impulsivity. Crackle is a
sequence of transients, so inside a short window one source frequently dominates
outright. Running a single-source search on many short windows and clustering
the maxima therefore recovers the sources without any deflation at all, which
makes the count it reports a genuine cross-check rather than a restatement of
the sequential loop.
"""

from dataclasses import dataclass, replace

import numpy as np

from src.config.multi_source_localization_configuration import (
    MultiSourceLocalizationConfiguration,
)
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.spark.inverse.receiver_pair_index import (
    compute_maximum_absolute_lag_s,
    enumerate_receiver_pairs,
)
from src.spark.inverse.steered_response_power import run_coarse_to_fine_search
from src.spark.inverse.time_difference_of_arrival import (
    compute_pair_correlation_curves,
)
from src.utils.array_types import Float64Array


@dataclass(frozen=True)
class PositionCluster:
    """One dense group of per-window maxima, taken to be one source.

    Attributes:
        centroid_xy_m: Mean of the window maxima in the cluster.
        spread_m: Root-mean-square distance of those maxima from the centroid,
            an empirical uncertainty owing nothing to a noise model.
        window_count: How many windows landed in this cluster.
    """

    centroid_xy_m: Float64Array
    spread_m: float
    window_count: int


@dataclass(frozen=True)
class WindowedClustering:
    """Every cluster found, and the maxima they were formed from.

    Attributes:
        clusters: Clusters with at least the minimum membership, largest first.
        estimated_source_count: Number of such clusters.
        window_maxima_xy_m: Every per-window maximum, shape `(n_windows, 2)`.
    """

    clusters: list[PositionCluster]
    estimated_source_count: int
    window_maxima_xy_m: Float64Array


def cluster_positions_by_radius(
    positions_xy_m: Float64Array,
    clustering_radius_m: float,
    minimum_cluster_size: int,
) -> list[PositionCluster]:
    """Groups points that sit within one radius of a growing cluster centre.

    Args:
        positions_xy_m: Points to cluster, shape `(n_points, 2)`.
        clustering_radius_m: Distance within which points join a cluster.
        minimum_cluster_size: Smallest membership a cluster needs to be kept.

    Returns:
        The surviving clusters, most populous first.

    Raises:
        ValueError: If the radius is not positive.
    """
    if clustering_radius_m <= 0.0:
        raise ValueError("clustering_radius_m must be positive")
    points_xy_m = np.atleast_2d(np.asarray(positions_xy_m, dtype=np.float64))
    is_unassigned = np.ones(points_xy_m.shape[0], dtype=np.bool_)
    clusters: list[PositionCluster] = []

    while np.any(is_unassigned):
        remaining_indices = np.flatnonzero(is_unassigned)
        neighbour_counts = np.array(
            [
                int(
                    np.count_nonzero(
                        np.linalg.norm(
                            points_xy_m[remaining_indices] - points_xy_m[index], axis=1
                        )
                        <= clustering_radius_m
                    )
                )
                for index in remaining_indices
            ]
        )
        seed_index = remaining_indices[int(np.argmax(neighbour_counts))]
        member_mask = (
            np.linalg.norm(points_xy_m - points_xy_m[seed_index], axis=1)
            <= clustering_radius_m
        ) & is_unassigned
        members_xy_m = points_xy_m[member_mask]
        centroid_xy_m = members_xy_m.mean(axis=0)
        clusters.append(
            PositionCluster(
                centroid_xy_m=np.asarray(centroid_xy_m, dtype=np.float64),
                spread_m=float(
                    np.sqrt(
                        np.mean(np.sum((members_xy_m - centroid_xy_m) ** 2, axis=1))
                    )
                ),
                window_count=int(members_xy_m.shape[0]),
            )
        )
        is_unassigned &= ~member_mask

    return sorted(
        (
            cluster
            for cluster in clusters
            if cluster.window_count >= minimum_cluster_size
        ),
        key=lambda cluster: cluster.window_count,
        reverse=True,
    )


def estimate_positions_by_windowed_clustering(
    receiver_signals: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    sample_rate_hz: int,
    domain_extent_x_m: float,
    domain_extent_y_m: float,
    conditions: AtmosphericConditions,
    configuration: MultiSourceLocalizationConfiguration,
) -> WindowedClustering:
    """Runs a single-source search per short window and clusters the maxima.

    Args:
        receiver_signals: Channels, shape `(n_receivers, n_samples)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        sample_rate_hz: Sample rate in hertz.
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        conditions: Air conditions the estimator assumes.
        configuration: Correlation, map and clustering settings.

    Returns:
        The clusters found, with every per-window maximum behind them.

    Raises:
        ValueError: If the channels are shorter than one clustering window.
    """
    channels = np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    speed_of_sound_m_per_s = compute_speed_of_sound_m_per_s(
        conditions.air_temperature_celsius
    )
    receiver_pairs = enumerate_receiver_pairs(receivers.shape[0])
    maximum_absolute_lag_s = compute_maximum_absolute_lag_s(
        receivers,
        receiver_pairs,
        speed_of_sound_m_per_s,
        configuration.correlation.maximum_absolute_lag_margin,
    )

    window_sample_count = round(
        configuration.clustering.window_duration_s * sample_rate_hz
    )
    if channels.shape[1] < window_sample_count:
        raise ValueError("the channels are shorter than one clustering window")
    hop_sample_count = max(
        round(window_sample_count * (1.0 - configuration.clustering.window_overlap)), 1
    )
    window_correlation = replace(
        configuration.correlation,
        window_duration_s=min(
            configuration.correlation.window_duration_s,
            configuration.clustering.window_duration_s,
        ),
    )
    seed_correlation = replace(
        window_correlation,
        highest_frequency_hz=(
            configuration.steered_response_power.low_frequency_seed_hz
        ),
    )

    maxima_xy_m: list[Float64Array] = []
    for start in range(
        0, channels.shape[1] - window_sample_count + 1, hop_sample_count
    ):
        block = channels[:, start : start + window_sample_count]
        search = run_coarse_to_fine_search(
            compute_pair_correlation_curves(
                block,
                receiver_pairs,
                maximum_absolute_lag_s,
                sample_rate_hz,
                seed_correlation,
            ),
            compute_pair_correlation_curves(
                block,
                receiver_pairs,
                maximum_absolute_lag_s,
                sample_rate_hz,
                window_correlation,
            ),
            receivers,
            receiver_pairs,
            speed_of_sound_m_per_s,
            domain_extent_x_m,
            domain_extent_y_m,
            configuration.steered_response_power,
        )
        maxima_xy_m.append(search.position_xyz_m[:2])

    window_maxima_xy_m = np.stack(maxima_xy_m)
    clusters = cluster_positions_by_radius(
        window_maxima_xy_m,
        configuration.clustering.clustering_radius_m,
        configuration.clustering.minimum_cluster_size,
    )
    return WindowedClustering(
        clusters=clusters,
        estimated_source_count=len(clusters),
        window_maxima_xy_m=window_maxima_xy_m,
    )
