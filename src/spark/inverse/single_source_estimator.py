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
    alignment_shift_samples = int(round(arrival.delay_s * sample_rate_hz))

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
class SingleSourceLocalizer:
    receiver_1_xy_m: Float64Array
    receiver_2_xy_m: Float64Array
    conditions: AtmosphericConditions
    configuration: LocalizationConfiguration

    def localize(
        self,
        signal_1: Float64Array,
        signal_2: Float64Array,
        sample_rate_hz: int,
    ) -> SingleSourceLocalization:
        return localize_single_source(
            signal_1,
            signal_2,
            sample_rate_hz,
            self.receiver_1_xy_m,
            self.receiver_2_xy_m,
            self.conditions,
            self.configuration,
        )
