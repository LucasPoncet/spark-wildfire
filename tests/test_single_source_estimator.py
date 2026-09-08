from dataclasses import replace

import numpy as np
import pytest

from src.config.simulation_configuration import LocalizationConfiguration
from src.spark.acoustic.free_field_propagation import render_receiver_signals
from src.spark.acoustic.receiver_noise import add_white_noise_to_receiver_signals
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.spark.inverse.single_source_estimator import (
    SingleSourceLocalization,
    SingleSourceLocalizer,
    compute_error_ellipse_semi_axes_m,
    localize_single_source,
)
from src.utils.array_types import Float64Array

EXPECTED_WINDOW_COUNT: int = 19


def localize_truth(
    true_position_xy_m: Float64Array,
    atmosphere: AtmosphericConditions,
    localization: LocalizationConfiguration,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    sample_rate_hz: int,
    clip_duration_s: float,
    reference_distance_m: float,
    signal_to_noise_ratio_db: float | None = None,
    seed: int = 0,
) -> SingleSourceLocalization:
    source_signal = np.random.default_rng(seed).normal(
        size=int(clip_duration_s * sample_rate_hz)
    )
    receiver_signals = render_receiver_signals(
        source_signal,
        sample_rate_hz,
        true_position_xy_m,
        np.vstack([receiver_1_xy_m, receiver_2_xy_m]),
        atmosphere,
        reference_distance_m,
    )
    if signal_to_noise_ratio_db is not None:
        receiver_signals = add_white_noise_to_receiver_signals(
            receiver_signals,
            signal_to_noise_ratio_db,
            np.random.default_rng(seed + 100),
        )
    return localize_single_source(
        receiver_signals[0],
        receiver_signals[1],
        sample_rate_hz,
        receiver_1_xy_m,
        receiver_2_xy_m,
        atmosphere,
        localization,
    )


@pytest.fixture
def localize(
    atmosphere: AtmosphericConditions,
    localization: LocalizationConfiguration,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    sample_rate_hz: int,
    clip_duration_s: float,
    reference_distance_m: float,
):
    def run(
        true_position_xy_m: Float64Array,
        signal_to_noise_ratio_db: float | None = None,
        seed: int = 0,
    ) -> SingleSourceLocalization:
        return localize_truth(
            true_position_xy_m,
            atmosphere,
            localization,
            receiver_1_xy_m,
            receiver_2_xy_m,
            sample_rate_hz,
            clip_duration_s,
            reference_distance_m,
            signal_to_noise_ratio_db,
            seed,
        )

    return run


@pytest.mark.parametrize(
    "true_position_xy_m",
    [
        np.array([25.0, 70.0]),
        np.array([75.0, 35.0]),
        np.array([40.0, 20.0]),
        np.array([60.0, 85.0]),
    ],
)
def test_noiseless_synthetic_source_is_recovered(
    localize, true_position_xy_m: Float64Array
) -> None:
    localization = localize(true_position_xy_m)
    assert np.linalg.norm(localization.position_xy_m - true_position_xy_m) < 0.5


def test_path_difference_matches_the_true_geometry(
    localize, receiver_1_xy_m: Float64Array, receiver_2_xy_m: Float64Array
) -> None:
    true_position_xy_m = np.array([25.0, 70.0])
    result = localize(true_position_xy_m)
    expected_path_difference_m = float(
        np.linalg.norm(true_position_xy_m - receiver_2_xy_m)
        - np.linalg.norm(true_position_xy_m - receiver_1_xy_m)
    )
    assert result.path_difference_m == pytest.approx(
        expected_path_difference_m, abs=0.05
    )


def test_geometric_level_difference_matches_the_true_range_ratio(
    localize, receiver_1_xy_m: Float64Array, receiver_2_xy_m: Float64Array
) -> None:
    true_position_xy_m = np.array([40.0, 20.0])
    result = localize(true_position_xy_m)
    expected_level_difference_db = 20.0 * np.log10(
        np.linalg.norm(true_position_xy_m - receiver_2_xy_m)
        / np.linalg.norm(true_position_xy_m - receiver_1_xy_m)
    )
    assert result.geometric_level_difference_db == pytest.approx(
        expected_level_difference_db, abs=0.05
    )


def test_window_count_matches_the_configured_plan(localize) -> None:
    assert localize(np.array([40.0, 20.0])).window_count == EXPECTED_WINDOW_COUNT


def test_source_on_the_perpendicular_bisector_is_flagged(
    localize, receiver_1_xy_m: Float64Array, receiver_2_xy_m: Float64Array
) -> None:
    midpoint = 0.5 * (receiver_1_xy_m + receiver_2_xy_m)
    assert localize(midpoint + np.array([0.0, 50.0])).is_near_singular


def test_source_on_the_perpendicular_bisector_collapses_onto_the_baseline(
    localize, receiver_1_xy_m: Float64Array, receiver_2_xy_m: Float64Array
) -> None:
    midpoint = 0.5 * (receiver_1_xy_m + receiver_2_xy_m)
    result = localize(midpoint + np.array([0.0, 50.0]))
    assert result.path_difference_m == pytest.approx(0.0, abs=1e-6)
    assert result.range_receiver_1_m == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(result.position_xy_m, midpoint, atol=1e-6)


def test_error_grows_with_receiver_noise(localize) -> None:
    true_position_xy_m = np.array([25.0, 70.0])
    errors_m = [
        float(
            np.mean(
                [
                    np.linalg.norm(
                        localize(true_position_xy_m, snr_db, seed).position_xy_m
                        - true_position_xy_m
                    )
                    for seed in range(4)
                ]
            )
        )
        for snr_db in [30.0, 10.0, 0.0]
    ]
    assert errors_m[0] < errors_m[1] < errors_m[2]


def test_ellipse_grows_with_receiver_noise(localize) -> None:
    true_position_xy_m = np.array([25.0, 70.0])
    semi_major_m = [
        float(
            np.mean(
                [
                    compute_error_ellipse_semi_axes_m(
                        localize(
                            true_position_xy_m, snr_db, seed
                        ).position_covariance_m2
                    )[0]
                    for seed in range(3)
                ]
            )
        )
        for snr_db in [30.0, 10.0, 0.0]
    ]
    assert semi_major_m[0] < semi_major_m[1] < semi_major_m[2]


def test_ellipse_semi_axes_are_ordered(localize) -> None:
    result = localize(np.array([25.0, 70.0]), 10.0)
    semi_major_m, semi_minor_m, _ = compute_error_ellipse_semi_axes_m(
        result.position_covariance_m2
    )
    assert semi_major_m >= semi_minor_m >= 0.0


def test_localizer_class_matches_the_functional_interface(
    atmosphere: AtmosphericConditions,
    localization: LocalizationConfiguration,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    sample_rate_hz: int,
    clip_duration_s: float,
    reference_distance_m: float,
) -> None:
    true_position_xy_m = np.array([75.0, 35.0])
    source_signal = np.random.default_rng(0).normal(
        size=int(clip_duration_s * sample_rate_hz)
    )
    receiver_signals = render_receiver_signals(
        source_signal,
        sample_rate_hz,
        true_position_xy_m,
        np.vstack([receiver_1_xy_m, receiver_2_xy_m]),
        atmosphere,
        reference_distance_m,
    )
    localizer = SingleSourceLocalizer(
        receiver_1_xy_m, receiver_2_xy_m, atmosphere, localization
    )
    from_class = localizer.localize(
        receiver_signals[0], receiver_signals[1], sample_rate_hz
    )
    from_function = localize_single_source(
        receiver_signals[0],
        receiver_signals[1],
        sample_rate_hz,
        receiver_1_xy_m,
        receiver_2_xy_m,
        atmosphere,
        localization,
    )
    assert np.allclose(from_class.position_xy_m, from_function.position_xy_m)


def test_swapping_the_receivers_mirrors_the_estimate_across_the_baseline(
    atmosphere: AtmosphericConditions,
    localization: LocalizationConfiguration,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    sample_rate_hz: int,
    clip_duration_s: float,
    reference_distance_m: float,
) -> None:
    true_position_xy_m = np.array([40.0, 20.0])
    source_signal = np.random.default_rng(0).normal(
        size=int(clip_duration_s * sample_rate_hz)
    )
    receiver_signals = render_receiver_signals(
        source_signal,
        sample_rate_hz,
        true_position_xy_m,
        np.vstack([receiver_1_xy_m, receiver_2_xy_m]),
        atmosphere,
        reference_distance_m,
    )
    mirrored_configuration = replace(
        localization,
        triangulation=replace(
            localization.triangulation,
            domain_side_sign=-localization.triangulation.domain_side_sign,
        ),
    )
    swapped = localize_single_source(
        receiver_signals[1],
        receiver_signals[0],
        sample_rate_hz,
        receiver_2_xy_m,
        receiver_1_xy_m,
        atmosphere,
        mirrored_configuration,
    )
    assert np.allclose(swapped.position_xy_m, true_position_xy_m, atol=0.5)


def test_band_diagnostics_have_one_entry_per_configured_band(
    localize, localization: LocalizationConfiguration
) -> None:
    result = localize(np.array([40.0, 20.0]))
    band_count = len(localization.bands.center_frequencies_hz)
    assert result.band_center_frequencies_hz.size == band_count
    assert result.band_geometric_level_difference_db.size == band_count
    assert result.band_window_variance_db2.size == band_count
    assert result.band_mean_variance_db2.size == band_count
    assert result.band_residual_db.size == band_count
    assert result.absorption_coefficients_db_per_m.size == band_count
