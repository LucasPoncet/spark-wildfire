from dataclasses import dataclass

import numpy as np

from src.config.simulation_configuration import LocalizationConfiguration
from src.spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.spark.inverse.band_level_difference import compute_band_level_differences
from src.spark.inverse.inverse_variance_fusion import fuse_inverse_variance
from src.spark.inverse.level_ratio_triangulation import triangulate_from_level_ratio
from src.spark.inverse.time_difference_of_arrival import (
    estimate_time_difference_of_arrival,
)
from src.utils.array_types import Float64Array


@dataclass(frozen=True)
class SingleSourceLocalization:
    """A source position with every diagnostic needed to judge it.

    Attributes:
        position_xy_m: Estimated position `(x, y)` in metres.
        position_covariance_m2: Two-by-two position covariance in metres squared.
        range_receiver_1_m: Estimated range to receiver 1, in metres.
        range_receiver_2_m: Estimated range to receiver 2, in metres.
        path_difference_m: Range difference from the delay, in metres.
        path_difference_variance_m2: Variance of that difference.
        time_difference_of_arrival_s: Delay in seconds, positive when receiver 1
            hears the event later.
        geometric_level_difference_db: Fused level difference in decibels.
        geometric_level_difference_variance_db2: Variance of that fused value.
        reduced_chi_square: About 1 when the bands agree; much larger means the
            free-field model does not hold and the position should not be trusted.
        band_center_frequencies_hz: ISO band centres.
        band_geometric_level_difference_db: Per-band level difference.
        band_window_variance_db2: Per-band variance across windows.
        band_mean_variance_db2: Per-band variance of the mean.
        band_residual_db: Per-band departure from the fused value. A rise with
            frequency points at the absorption coefficients, a dip at 250 to 500 Hz
            at ground reflection.
        absorption_coefficients_db_per_m: Coefficient assumed for each band.
        speed_of_sound_m_per_s: Speed of sound assumed for the scenario.
        window_count: Number of analysis windows used.
        effective_window_count: Number of windows counted as independent.
        magnitude_squared_coherence: Mean coherence between the aligned channels.
        is_near_singular: Whether the source lies near the perpendicular bisector.
    """

    position_xy_m: Float64Array
    position_covariance_m2: Float64Array
    range_receiver_1_m: float
    range_receiver_2_m: float
    path_difference_m: float
    path_difference_variance_m2: float
    time_difference_of_arrival_s: float
    geometric_level_difference_db: float
    geometric_level_difference_variance_db2: float
    reduced_chi_square: float
    band_center_frequencies_hz: Float64Array
    band_geometric_level_difference_db: Float64Array
    band_window_variance_db2: Float64Array
    band_mean_variance_db2: Float64Array
    band_residual_db: Float64Array
    absorption_coefficients_db_per_m: Float64Array
    speed_of_sound_m_per_s: float
    window_count: int
    effective_window_count: float
    magnitude_squared_coherence: float
    is_near_singular: bool


def compute_error_ellipse_semi_axes_m(
    position_covariance_m2: Float64Array,
) -> tuple[float, float, float]:
    """Turns a position covariance into a one-sigma error ellipse.

    Args:
        position_covariance_m2: Two-by-two covariance in metres squared.

    Returns:
        `(semi_major_m, semi_minor_m, orientation_rad)`, the orientation measured
        from the x-axis to the major axis.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.asarray(position_covariance_m2, dtype=np.float64)
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    major_direction = eigenvectors[:, order[0]]
    return (
        float(np.sqrt(eigenvalues[0])),
        float(np.sqrt(eigenvalues[1])),
        float(np.arctan2(major_direction[1], major_direction[0])),
    )


def localize_single_source(
    signal_1: Float64Array,
    signal_2: Float64Array,
    sample_rate_hz: int,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    conditions: AtmosphericConditions,
    configuration: LocalizationConfiguration,
) -> SingleSourceLocalization:
    """Locates one source from two receiver signals and nothing else.

    Fuses the per-band evidence into a single geometric term first, then inverts
    once. The delay fixes the range difference, the fused level ratio fixes the
    range ratio, and the two together fix the position.

    Args:
        signal_1: Receiver 1 channel.
        signal_2: Receiver 2 channel, same clock and sample rate.
        sample_rate_hz: Sample rate in hertz.
        receiver_1_xy_m: Receiver 1 position `(x, y)` in metres.
        receiver_2_xy_m: Receiver 2 position `(x, y)` in metres.
        conditions: Air conditions the estimator assumes, supplied by the caller and
            never read back from a simulation.
        configuration: Bands, window, delay estimation and triangulation settings.

    Returns:
        The estimated position with its covariance and per-band diagnostics.
    """
    receiver_1 = np.asarray(receiver_1_xy_m, dtype=np.float64)
    receiver_2 = np.asarray(receiver_2_xy_m, dtype=np.float64)
    speed_of_sound_m_per_s = compute_speed_of_sound_m_per_s(
        conditions.air_temperature_celsius
    )
    baseline_m = float(np.linalg.norm(receiver_2 - receiver_1))

    arrival = estimate_time_difference_of_arrival(
        signal_1,
        signal_2,
        sample_rate_hz,
        baseline_m / speed_of_sound_m_per_s,
        configuration.delay_estimation,
    )
    path_difference_m = -speed_of_sound_m_per_s * arrival.delay_s
    path_difference_variance_m2 = speed_of_sound_m_per_s**2 * arrival.variance_s2
    alignment_shift_samples = round(arrival.delay_s * sample_rate_hz)

    absorption_coefficients_db_per_m = compute_absorption_coefficients_db_per_m(
        np.asarray(configuration.bands.center_frequencies_hz, dtype=np.float64),
        conditions,
    )
    band_differences = compute_band_level_differences(
        signal_1,
        signal_2,
        sample_rate_hz,
        absorption_coefficients_db_per_m,
        path_difference_m,
        alignment_shift_samples,
        configuration.bands,
        configuration.window,
    )
    fused = fuse_inverse_variance(
        band_differences.geometric_level_difference_db,
        band_differences.mean_variance_db2,
    )
    triangulated = triangulate_from_level_ratio(
        fused.value,
        path_difference_m,
        fused.variance,
        path_difference_variance_m2,
        receiver_1,
        receiver_2,
        configuration.triangulation,
    )

    return SingleSourceLocalization(
        position_xy_m=triangulated.position_xy_m,
        position_covariance_m2=triangulated.position_covariance_m2,
        range_receiver_1_m=triangulated.range_receiver_1_m,
        range_receiver_2_m=triangulated.range_receiver_2_m,
        path_difference_m=path_difference_m,
        path_difference_variance_m2=path_difference_variance_m2,
        time_difference_of_arrival_s=arrival.delay_s,
        geometric_level_difference_db=fused.value,
        geometric_level_difference_variance_db2=fused.variance,
        reduced_chi_square=fused.reduced_chi_square,
        band_center_frequencies_hz=band_differences.band_center_frequencies_hz,
        band_geometric_level_difference_db=band_differences.geometric_level_difference_db,
        band_window_variance_db2=band_differences.window_variance_db2,
        band_mean_variance_db2=band_differences.mean_variance_db2,
        band_residual_db=fused.residuals,
        absorption_coefficients_db_per_m=absorption_coefficients_db_per_m,
        speed_of_sound_m_per_s=speed_of_sound_m_per_s,
        window_count=band_differences.window_count,
        effective_window_count=band_differences.effective_window_count,
        magnitude_squared_coherence=arrival.magnitude_squared_coherence,
        is_near_singular=triangulated.is_near_singular,
    )


@dataclass(frozen=True)
class SingleSourceEstimator:
    """Locates one fire front from a fixed pair of receivers.

    Holds the receiver geometry and the assumed air conditions so a run can
    localize clip after clip through one call.

    Note that this estimator consumes receiver *waveforms*, not the scalar
    `receiver_signal_levels` of `KinematicsEstimatorProtocol`: the range difference
    comes from a time delay, which levels alone cannot supply.

    Attributes:
        receiver_1_xy_m: Receiver 1 position `(x, y)` in metres.
        receiver_2_xy_m: Receiver 2 position `(x, y)` in metres.
        conditions: Air conditions the estimator assumes.
        configuration: Bands, window, delay estimation and triangulation settings.
    """

    receiver_1_xy_m: Float64Array
    receiver_2_xy_m: Float64Array
    conditions: AtmosphericConditions
    configuration: LocalizationConfiguration

    def estimate(
        self,
        signal_1: Float64Array,
        signal_2: Float64Array,
        sample_rate_hz: int,
    ) -> SingleSourceLocalization:
        """Locates the source behind one pair of receiver signals.

        Args:
            signal_1: Receiver 1 channel.
            signal_2: Receiver 2 channel, same clock and sample rate.
            sample_rate_hz: Sample rate in hertz.

        Returns:
            The estimated position with its covariance and per-band diagnostics.
        """
        return localize_single_source(
            signal_1,
            signal_2,
            sample_rate_hz,
            self.receiver_1_xy_m,
            self.receiver_2_xy_m,
            self.conditions,
            self.configuration,
        )
