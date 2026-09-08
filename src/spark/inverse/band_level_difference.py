from dataclasses import dataclass

import numpy as np

from src.audio.band_level_meter import (
    compute_band_level_db,
    compute_window_start_indices,
    make_analysis_window,
)
from src.audio.octave_band_filter import apply_octave_bandpass
from src.audio.signal_alignment import compute_valid_index_range, shift_signal_by_samples
from src.config.simulation_configuration import BandConfiguration, WindowConfiguration
from src.utils.array_types import Float64Array


@dataclass(frozen=True)
class BandLevelDifferences:
    band_center_frequencies_hz: Float64Array
    geometric_level_difference_db: Float64Array
    window_variance_db2: Float64Array
    mean_variance_db2: Float64Array
    window_count: int
    effective_window_count: float


def compute_effective_window_count(
    window_count: int, window_configuration: WindowConfiguration
) -> float:
    if window_configuration.overlap_fraction <= 0.0:
        return float(window_count)
    return window_configuration.independent_window_fraction * window_count


def compute_band_level_differences(
    signal_1: Float64Array,
    signal_2: Float64Array,
    sample_rate_hz: int,
    absorption_coefficients_db_per_m: Float64Array,
    path_difference_m: float,
    alignment_shift_samples: int,
    band_configuration: BandConfiguration,
    window_configuration: WindowConfiguration,
) -> BandLevelDifferences:
    first = np.asarray(signal_1, dtype=np.float64)
    second = np.asarray(signal_2, dtype=np.float64)
    centers_hz = np.asarray(band_configuration.center_frequencies_hz, dtype=np.float64)
    absorption = np.asarray(absorption_coefficients_db_per_m, dtype=np.float64)
    if absorption.size != centers_hz.size:
        raise ValueError("one absorption coefficient is required per octave band")

    window_sample_count = int(round(window_configuration.duration_s * sample_rate_hz))
    hop_sample_count = max(
        int(round(window_sample_count * (1.0 - window_configuration.overlap_fraction))), 1
    )
    window = make_analysis_window(window_sample_count)
    first_index, last_index = compute_valid_index_range(
        first.size, second.size, alignment_shift_samples
    )
    window_starts = compute_window_start_indices(
        first_index, last_index, window_sample_count, hop_sample_count
    )
    if len(window_starts) < 2:
        raise ValueError("at least two analysis windows are required to estimate a variance")

    window_count = len(window_starts)
    effective_window_count = compute_effective_window_count(window_count, window_configuration)

    means_db: list[float] = []
    window_variances_db2: list[float] = []
    for center_frequency_hz, absorption_db_per_m in zip(centers_hz, absorption, strict=True):
        band_1 = apply_octave_bandpass(
            first,
            float(center_frequency_hz),
            sample_rate_hz,
            band_configuration.filter_order,
            band_configuration.maximum_edge_fraction_of_nyquist,
        )
        band_2 = shift_signal_by_samples(
            apply_octave_bandpass(
                second,
                float(center_frequency_hz),
                sample_rate_hz,
                band_configuration.filter_order,
                band_configuration.maximum_edge_fraction_of_nyquist,
            ),
            alignment_shift_samples,
        )
        estimates_db = np.array(
            [
                compute_band_level_db(band_1[start : start + window_sample_count], window)
                - compute_band_level_db(band_2[start : start + window_sample_count], window)
                - absorption_db_per_m * path_difference_m
                for start in window_starts
            ],
            dtype=np.float64,
        )
        means_db.append(float(estimates_db.mean()))
        window_variances_db2.append(
            float(estimates_db.var(ddof=1) + window_configuration.variance_floor_db2)
        )

    window_variance_db2 = np.array(window_variances_db2, dtype=np.float64)
    return BandLevelDifferences(
        band_center_frequencies_hz=centers_hz,
        geometric_level_difference_db=np.array(means_db, dtype=np.float64),
        window_variance_db2=window_variance_db2,
        mean_variance_db2=window_variance_db2 / effective_window_count,
        window_count=window_count,
        effective_window_count=effective_window_count,
    )
