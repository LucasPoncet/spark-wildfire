"""Locating several concurrent fire fronts from receiver waveforms alone.

Structured as the X-SRP loop: build the pair correlations once, search the
domain for the strongest candidate, refine it against every pair delay, peel it
out of the correlations, and search again. Deflation is `update_signal_features`
and coarse-to-fine refinement is `update_grid`, so one loop covers both the
search and the peeling.

The source count is an output, never an input. Where the loop stops is decided
by the prominence of the residual map peak, the separation from what is already
accepted, and whether peeling the last source actually removed any correlation
energy.
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
from src.spark.inverse.multilateration import (
    PositionEstimate,
    compute_predicted_time_differences_s,
    refine_position_gauss_newton,
)
from src.spark.inverse.receiver_pair_index import (
    compute_maximum_absolute_lag_s,
    enumerate_receiver_pairs,
)
from src.spark.inverse.source_deflation import (
    compute_residual_correlation_energy_ratio,
    deflate_located_source,
)
from src.spark.inverse.steered_response_power import (
    SteeredResponseSearch,
    run_coarse_to_fine_search,
)
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
    compute_pair_correlation_curves,
    estimate_delay_near_prediction,
)
from src.utils.array_types import Float64Array, Int64Array

DELAY_SEARCH_CELL_FACTOR: float = 4.0
DELAY_VARIANCE_INTERPOLATION_FACTOR: int = 16

ACCEPTED_MAXIMUM_SOURCE_COUNT: str = "maximum_source_count_reached"
STOPPED_ON_PROMINENCE: str = "map_peak_not_prominent"
STOPPED_ON_SEPARATION: str = "peak_too_close_to_an_accepted_source"
STOPPED_ON_RESIDUAL: str = "deflation_stopped_removing_energy"


@dataclass(frozen=True)
class LocatedSource:
    """One source the loop accepted, with the evidence behind it.

    Attributes:
        position: Refined position with its covariance and residuals.
        map_position_xyz_m: Position the map peak sat at before refinement.
        map_peak_value: Value of the map at that peak.
        map_peak_to_median_ratio: How far that peak stood above the map median.
        residual_energy_ratio: Share of correlation energy still present after
            this source was peeled out.
        pair_delays_s: Delay attributed to this source on each pair.
        pair_delay_variances_s2: Variance of each of those delays.
    """

    position: PositionEstimate
    map_position_xyz_m: Float64Array
    map_peak_value: float
    map_peak_to_median_ratio: float
    residual_energy_ratio: float
    pair_delays_s: Float64Array
    pair_delay_variances_s2: Float64Array


@dataclass(frozen=True)
class MultiSourceLocalization:
    """Every source the loop accepted, and why it stopped.

    Attributes:
        sources: Accepted sources, strongest first.
        estimated_source_count: How many sources were accepted.
        receiver_positions_xyz_m: The array the estimate came from.
        receiver_pairs: Canonical pairs used.
        speed_of_sound_m_per_s: Speed of sound the estimator assumed.
        stop_reason: Which condition ended the loop.
        first_round_map: Coarse map of the first round, kept for display.
        first_round_positions_xyz_m: Cell centres of that map.
    """

    sources: list[LocatedSource]
    estimated_source_count: int
    receiver_positions_xyz_m: Float64Array
    receiver_pairs: Int64Array
    speed_of_sound_m_per_s: float
    stop_reason: str
    first_round_map: Float64Array
    first_round_positions_xyz_m: Float64Array


def is_far_enough_from_accepted(
    candidate_position_xyz_m: Float64Array,
    accepted_sources: list[LocatedSource],
    minimum_separation_m: float,
) -> bool:
    """Reports whether a candidate is a new source rather than a re-detection.

    Args:
        candidate_position_xyz_m: The candidate, shape `(3,)`.
        accepted_sources: Sources already accepted.
        minimum_separation_m: Closest an accepted source may sit to another.

    Returns:
        True when the candidate is far enough from every accepted source.
    """
    candidate_xy_m = np.asarray(candidate_position_xyz_m, dtype=np.float64)[:2]
    return all(
        float(np.linalg.norm(candidate_xy_m - source.position.position_xy_m))
        >= minimum_separation_m
        for source in accepted_sources
    )


def read_pair_delays_for_source(
    correlation_curves: list[GeneralizedCrossCorrelationCurve],
    source_position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    search_half_width_s: float,
    observation_duration_s: float,
    peak_interpolation: str,
) -> tuple[Float64Array, Float64Array]:
    """Reads the delay each pair sees from one located source.

    Args:
        correlation_curves: One curve per receiver pair.
        source_position_xyz_m: The located source, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        search_half_width_s: How far either side of the prediction to look.
        observation_duration_s: Length of the record the curves came from.
        peak_interpolation: Sub-sample refinement rule.

    Returns:
        `(delays_s, variances_s2)`, both of shape `(n_pairs,)`.
    """
    predicted_delays_s = compute_predicted_time_differences_s(
        source_position_xyz_m,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
    )
    estimates = [
        estimate_delay_near_prediction(
            curve,
            float(predicted_delay_s),
            search_half_width_s,
            peak_interpolation,
            observation_duration_s,
            DELAY_VARIANCE_INTERPOLATION_FACTOR,
        )
        for curve, predicted_delay_s in zip(
            correlation_curves, predicted_delays_s, strict=True
        )
    ]
    return (
        np.array([estimate.delay_s for estimate in estimates], dtype=np.float64),
        np.array([estimate.variance_s2 for estimate in estimates], dtype=np.float64),
    )


def localize_multiple_sources(
    receiver_signals: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    sample_rate_hz: int,
    domain_extent_x_m: float,
    domain_extent_y_m: float,
    conditions: AtmosphericConditions,
    configuration: MultiSourceLocalizationConfiguration,
) -> MultiSourceLocalization:
    """Locates every source it can find, without being told how many there are.

    Args:
        receiver_signals: Channels, shape `(n_receivers, n_samples)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        sample_rate_hz: Sample rate in hertz.
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        conditions: Air conditions the estimator assumes, supplied by the caller
            and never read back from a simulation.
        configuration: Correlation, map, deflation and refinement settings.

    Returns:
        The accepted sources and the condition that ended the search.
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
    observation_duration_s = channels.shape[1] / sample_rate_hz

    correlation_curves = compute_pair_correlation_curves(
        channels,
        receiver_pairs,
        maximum_absolute_lag_s,
        sample_rate_hz,
        configuration.correlation,
    )
    seed_configuration = replace(
        configuration.correlation,
        highest_frequency_hz=(
            configuration.steered_response_power.low_frequency_seed_hz
        ),
    )
    seed_curves = compute_pair_correlation_curves(
        channels,
        receiver_pairs,
        maximum_absolute_lag_s,
        sample_rate_hz,
        seed_configuration,
    )

    accepted: list[LocatedSource] = []
    stop_reason = ACCEPTED_MAXIMUM_SOURCE_COUNT
    first_search: SteeredResponseSearch | None = None

    while len(accepted) < configuration.deflation.maximum_source_count:
        search = run_coarse_to_fine_search(
            seed_curves,
            correlation_curves,
            receivers,
            receiver_pairs,
            speed_of_sound_m_per_s,
            domain_extent_x_m,
            domain_extent_y_m,
            configuration.steered_response_power,
        )
        if first_search is None:
            first_search = search
        if (
            search.coarse_peak_to_median_ratio
            < configuration.deflation.peak_prominence_threshold_ratio
        ):
            stop_reason = STOPPED_ON_PROMINENCE
            break
        if not is_far_enough_from_accepted(
            search.position_xyz_m,
            accepted,
            configuration.deflation.minimum_source_separation_m,
        ):
            stop_reason = STOPPED_ON_SEPARATION
            break

        delays_s, variances_s2 = read_pair_delays_for_source(
            correlation_curves,
            search.position_xyz_m,
            receivers,
            receiver_pairs,
            speed_of_sound_m_per_s,
            DELAY_SEARCH_CELL_FACTOR
            * search.final_cell_extent_m
            / speed_of_sound_m_per_s,
            observation_duration_s,
            configuration.correlation.peak_interpolation,
        )
        position = refine_position_gauss_newton(
            search.position_xyz_m,
            delays_s,
            variances_s2,
            receivers,
            receiver_pairs,
            speed_of_sound_m_per_s,
            configuration.refinement,
        )

        deflated_curves = deflate_located_source(
            correlation_curves,
            search.position_xyz_m,
            receivers,
            receiver_pairs,
            speed_of_sound_m_per_s,
            configuration.deflation.method,
            configuration.deflation.notch_width_s,
        )
        residual_energy_ratio = compute_residual_correlation_energy_ratio(
            correlation_curves, deflated_curves
        )
        if (
            1.0 - residual_energy_ratio
            < configuration.deflation.residual_power_reduction_threshold
        ):
            stop_reason = STOPPED_ON_RESIDUAL
            break

        accepted.append(
            LocatedSource(
                position=position,
                map_position_xyz_m=search.position_xyz_m,
                map_peak_value=search.peak_value,
                map_peak_to_median_ratio=search.coarse_peak_to_median_ratio,
                residual_energy_ratio=residual_energy_ratio,
                pair_delays_s=delays_s,
                pair_delay_variances_s2=variances_s2,
            )
        )
        correlation_curves = deflated_curves
        seed_curves = deflate_located_source(
            seed_curves,
            search.position_xyz_m,
            receivers,
            receiver_pairs,
            speed_of_sound_m_per_s,
            configuration.deflation.method,
            configuration.deflation.notch_width_s,
        )

    return MultiSourceLocalization(
        sources=accepted,
        estimated_source_count=len(accepted),
        receiver_positions_xyz_m=receivers,
        receiver_pairs=receiver_pairs,
        speed_of_sound_m_per_s=speed_of_sound_m_per_s,
        stop_reason=stop_reason,
        first_round_map=(
            first_search.coarse_map
            if first_search is not None
            else np.empty(0, dtype=np.float64)
        ),
        first_round_positions_xyz_m=(
            first_search.coarse_positions_xyz_m
            if first_search is not None
            else np.empty((0, 3), dtype=np.float64)
        ),
    )


@dataclass(frozen=True)
class MultipleSourceEstimator:
    """Locates several fire fronts from one fixed receiver array.

    Holds the geometry and the assumed air conditions so a run can localize clip
    after clip through one call. Like `SingleSourceEstimator` it consumes
    receiver waveforms rather than scalar levels, because the geometry it
    recovers lives in the delays.

    Attributes:
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        conditions: Air conditions the estimator assumes.
        configuration: Correlation, map, deflation and refinement settings.
    """

    receiver_positions_xyz_m: Float64Array
    domain_extent_x_m: float
    domain_extent_y_m: float
    conditions: AtmosphericConditions
    configuration: MultiSourceLocalizationConfiguration

    def estimate(
        self, receiver_signals: Float64Array, sample_rate_hz: int
    ) -> MultiSourceLocalization:
        """Locates every source behind one set of receiver signals.

        Args:
            receiver_signals: Channels, shape `(n_receivers, n_samples)`.
            sample_rate_hz: Sample rate in hertz.

        Returns:
            The accepted sources and the condition that ended the search.
        """
        return localize_multiple_sources(
            receiver_signals,
            self.receiver_positions_xyz_m,
            sample_rate_hz,
            self.domain_extent_x_m,
            self.domain_extent_y_m,
            self.conditions,
            self.configuration,
        )
