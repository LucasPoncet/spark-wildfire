from dataclasses import replace

import numpy as np
import pytest

from src.config.simulation_configuration import BandConfiguration, WindowConfiguration
from src.spark.acoustic.free_field_propagation import render_receiver_signals
from src.spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.spark.inverse.band_level_difference import (
    BandLevelDifferences,
    compute_band_level_differences,
    compute_effective_window_count,
)
from src.utils.array_types import Float64Array

TRUE_POSITION_XY_M = np.array([25.0, 70.0])


@pytest.fixture
def absorption_coefficients_db_per_m(
    atmosphere: AtmosphericConditions, bands: BandConfiguration
) -> Float64Array:
    return compute_absorption_coefficients_db_per_m(
        np.asarray(bands.center_frequencies_hz, dtype=np.float64), atmosphere
    )


@pytest.fixture
def true_ranges_m(
    receiver_1_xy_m: Float64Array, receiver_2_xy_m: Float64Array
) -> tuple[float, float]:
    return (
        float(np.linalg.norm(TRUE_POSITION_XY_M - receiver_1_xy_m)),
        float(np.linalg.norm(TRUE_POSITION_XY_M - receiver_2_xy_m)),
    )


@pytest.fixture
def band_differences(
    atmosphere: AtmosphericConditions,
    bands: BandConfiguration,
    window: WindowConfiguration,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    sample_rate_hz: int,
    clip_duration_s: float,
    reference_distance_m: float,
    absorption_coefficients_db_per_m: Float64Array,
    true_ranges_m: tuple[float, float],
) -> BandLevelDifferences:
    source_signal = np.random.default_rng(0).normal(
        size=int(clip_duration_s * sample_rate_hz)
    )
    receiver_signals = render_receiver_signals(
        source_signal,
        sample_rate_hz,
        TRUE_POSITION_XY_M,
        np.vstack([receiver_1_xy_m, receiver_2_xy_m]),
        atmosphere,
        reference_distance_m,
    )
    range_1_m, range_2_m = true_ranges_m
    return compute_band_level_differences(
        receiver_signals[0],
        receiver_signals[1],
        sample_rate_hz,
        absorption_coefficients_db_per_m,
        range_2_m - range_1_m,
        0,
        bands,
        window,
    )


def test_non_overlapping_windows_all_count_as_independent() -> None:
    assert (
        compute_effective_window_count(19, WindowConfiguration(0.5, 0.0, 1e-6, 0.6))
        == 19.0
    )


def test_overlapping_windows_count_for_less() -> None:
    effective = compute_effective_window_count(
        19, WindowConfiguration(0.5, 0.5, 1e-6, 0.6)
    )
    assert effective == pytest.approx(0.6 * 19)


def test_one_estimate_per_configured_band(
    band_differences: BandLevelDifferences, bands: BandConfiguration
) -> None:
    band_count = len(bands.center_frequencies_hz)
    assert band_differences.band_center_frequencies_hz.size == band_count
    assert band_differences.geometric_level_difference_db.size == band_count
    assert band_differences.window_variance_db2.size == band_count
    assert band_differences.mean_variance_db2.size == band_count


def test_variance_of_the_mean_is_below_the_window_variance(
    band_differences: BandLevelDifferences,
) -> None:
    assert np.all(
        band_differences.mean_variance_db2 < band_differences.window_variance_db2
    )


def test_effective_window_count_does_not_exceed_the_window_count(
    band_differences: BandLevelDifferences,
) -> None:
    assert band_differences.effective_window_count <= band_differences.window_count


def test_every_band_recovers_the_true_range_ratio(
    band_differences: BandLevelDifferences, true_ranges_m: tuple[float, float]
) -> None:
    range_1_m, range_2_m = true_ranges_m
    expected_db = 20.0 * np.log10(range_2_m / range_1_m)
    assert np.allclose(
        band_differences.geometric_level_difference_db, expected_db, atol=0.2
    )


def test_variances_are_floored_above_zero(
    band_differences: BandLevelDifferences, window: WindowConfiguration
) -> None:
    assert np.all(band_differences.window_variance_db2 >= window.variance_floor_db2)


def test_mismatched_absorption_count_is_rejected(
    bands: BandConfiguration, window: WindowConfiguration, sample_rate_hz: int
) -> None:
    signal = np.random.default_rng(0).normal(size=sample_rate_hz)
    with pytest.raises(ValueError, match="one absorption coefficient"):
        compute_band_level_differences(
            signal, signal, sample_rate_hz, np.array([0.001]), 10.0, 0, bands, window
        )


def test_a_clip_too_short_for_two_windows_is_rejected(
    bands: BandConfiguration,
    window: WindowConfiguration,
    sample_rate_hz: int,
    absorption_coefficients_db_per_m: Float64Array,
) -> None:
    short = np.random.default_rng(0).normal(size=sample_rate_hz // 4)
    long_window = replace(window, duration_s=1.0)
    with pytest.raises(ValueError, match="at least two analysis windows"):
        compute_band_level_differences(
            short,
            short,
            sample_rate_hz,
            absorption_coefficients_db_per_m,
            10.0,
            0,
            bands,
            long_window,
        )
