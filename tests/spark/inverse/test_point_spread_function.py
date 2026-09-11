"""The array's response to a single point source.

Two things here are load-bearing rather than descriptive. The first is the sign
convention: the synthesised curve must peak at the delay the steering table
predicts for the same position, because if it peaks at minus that, the response
is mirrored about the array and every covariance subtracted downstream is the
wrong one, silently. The second is that the covariance runs through the same
preparation a rendered frame does, since the extent stage subtracts one from
the other.
"""

from dataclasses import replace

import numpy as np
import pytest

from src.config.multi_source_localization_configuration import (
    CorrelationConfiguration,
    SteeredResponsePowerConfiguration,
)
from src.spark.inverse.point_spread_function import (
    build_point_spread_correlation_curves,
    build_point_spread_covariance_field,
    compute_point_spread_covariance,
    compute_point_spread_function,
    compute_point_spread_semi_axes_m,
    compute_whitened_autocorrelation_kernel,
)
from src.spark.inverse.receiver_pair_index import (
    compute_maximum_absolute_lag_s,
    enumerate_receiver_pairs,
)
from src.spark.inverse.steered_response_power import (
    build_candidate_grid_xyz,
    compute_pair_delay_table_s,
)

SAMPLE_RATE_HZ: int = 8000
DOMAIN_EXTENT_M: float = 24.0
GRID_SPACING_M: float = 1.0
SPEED_OF_SOUND_M_PER_S: float = 340.0
BACKGROUND_PERCENTILE: float = 60.0
CENTRE_XYZ_M = np.array([12.0, 12.0, 0.5])

RECEIVER_ANGLES_RAD = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
RECEIVERS_XYZ_M = np.stack(
    (
        12.0 + 10.0 * np.cos(RECEIVER_ANGLES_RAD),
        12.0 + 10.0 * np.sin(RECEIVER_ANGLES_RAD),
        np.full(6, 1.0),
    ),
    axis=1,
)
RECEIVER_PAIRS = enumerate_receiver_pairs(RECEIVERS_XYZ_M.shape[0])

CORRELATION = CorrelationConfiguration(
    lowest_frequency_hz=200.0,
    highest_frequency_hz=3500.0,
    phase_transform_exponent=1.0,
    phase_transform_regularization=1e-8,
    use_analytic_envelope=True,
    window_duration_s=0.1,
    window_overlap=0.5,
    maximum_absolute_lag_margin=1.1,
    peak_interpolation="exponential",
)
STEERED_RESPONSE_POWER = SteeredResponsePowerConfiguration(
    pooling="mean",
    pairwise_combinator="sum",
    coarse_grid_spacing_m=GRID_SPACING_M,
    refinement_levels=1,
    refinement_factor=2,
    refinement_retained_fraction=0.1,
    low_frequency_seed_hz=2000.0,
    candidate_height_m=0.5,
)
MAXIMUM_LAG_S = compute_maximum_absolute_lag_s(
    RECEIVERS_XYZ_M, RECEIVER_PAIRS, SPEED_OF_SOUND_M_PER_S, 1.1
)
GRID = build_candidate_grid_xyz(DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, GRID_SPACING_M, 0.5)


def build_response(reference_xyz_m: np.ndarray) -> np.ndarray:
    return compute_point_spread_function(
        reference_xyz_m,
        GRID,
        RECEIVERS_XYZ_M,
        RECEIVER_PAIRS,
        SPEED_OF_SOUND_M_PER_S,
        SAMPLE_RATE_HZ,
        MAXIMUM_LAG_S,
        CORRELATION,
        STEERED_RESPONSE_POWER,
        None,
    )


def build_covariance(
    reference_xyz_m: np.ndarray, correlation: CorrelationConfiguration
) -> np.ndarray:
    return compute_point_spread_covariance(
        reference_xyz_m,
        GRID,
        RECEIVERS_XYZ_M,
        RECEIVER_PAIRS,
        SPEED_OF_SOUND_M_PER_S,
        SAMPLE_RATE_HZ,
        MAXIMUM_LAG_S,
        correlation,
        STEERED_RESPONSE_POWER,
        None,
        BACKGROUND_PERCENTILE,
    )


def test_the_kernel_is_one_at_zero_lag() -> None:
    kernel = compute_whitened_autocorrelation_kernel(
        np.array([0.0]), 200.0, 3500.0, 1.0
    )
    assert float(kernel[0]) == pytest.approx(1.0)


def test_the_kernel_is_even_in_lag() -> None:
    lags_s = np.linspace(0.0, 2e-3, 201)
    assert compute_whitened_autocorrelation_kernel(
        lags_s, 200.0, 3500.0, 1.0
    ) == pytest.approx(
        compute_whitened_autocorrelation_kernel(-lags_s, 200.0, 3500.0, 1.0)
    )


def test_a_wider_band_gives_a_narrower_lobe() -> None:
    lags_s = np.linspace(-2e-3, 2e-3, 4001)
    narrow = compute_whitened_autocorrelation_kernel(lags_s, 200.0, 1000.0, 1.0)
    wide = compute_whitened_autocorrelation_kernel(lags_s, 200.0, 6000.0, 1.0)
    assert float(np.sum(narrow > 0.5)) > float(np.sum(wide > 0.5))


def test_the_exponent_does_not_change_a_flat_band_kernel() -> None:
    lags_s = np.linspace(-1e-3, 1e-3, 501)
    assert compute_whitened_autocorrelation_kernel(
        lags_s, 200.0, 3500.0, 0.0
    ) == pytest.approx(
        compute_whitened_autocorrelation_kernel(lags_s, 200.0, 3500.0, 1.0)
    )


def test_an_inverted_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive width"):
        compute_whitened_autocorrelation_kernel(np.array([0.0]), 3500.0, 200.0, 1.0)


def test_every_synthesised_curve_peaks_at_the_delay_the_steering_predicts() -> None:
    reference_xyz_m = np.array([15.0, 9.0, 0.5])
    curves = build_point_spread_correlation_curves(
        reference_xyz_m,
        RECEIVERS_XYZ_M,
        RECEIVER_PAIRS,
        SPEED_OF_SOUND_M_PER_S,
        SAMPLE_RATE_HZ,
        MAXIMUM_LAG_S,
        CORRELATION,
        None,
    )
    predicted_s = compute_pair_delay_table_s(
        reference_xyz_m.reshape(1, 3),
        RECEIVERS_XYZ_M,
        RECEIVER_PAIRS,
        SPEED_OF_SOUND_M_PER_S,
    )[0]
    peaks_s = np.array([curve.lags_s[int(np.argmax(curve.values))] for curve in curves])
    assert peaks_s == pytest.approx(predicted_s, abs=1.0 / SAMPLE_RATE_HZ)


def test_one_curve_is_produced_per_pair() -> None:
    curves = build_point_spread_correlation_curves(
        CENTRE_XYZ_M,
        RECEIVERS_XYZ_M,
        RECEIVER_PAIRS,
        SPEED_OF_SOUND_M_PER_S,
        SAMPLE_RATE_HZ,
        MAXIMUM_LAG_S,
        CORRELATION,
        None,
    )
    assert len(curves) == RECEIVER_PAIRS.shape[0]
    assert [curve.receiver_pair for curve in curves] == [
        (int(pair[0]), int(pair[1])) for pair in RECEIVER_PAIRS
    ]


def test_the_response_peaks_on_the_cell_the_source_sits_in() -> None:
    response = build_response(CENTRE_XYZ_M)
    peak_xy_m = GRID.positions_xyz_m[int(np.argmax(response)), :2]
    assert float(np.linalg.norm(peak_xy_m - CENTRE_XYZ_M[:2])) <= GRID_SPACING_M


def test_moving_the_source_moves_the_response_with_it() -> None:
    offset_xyz_m = np.array([16.0, 8.0, 0.5])
    peak_xy_m = GRID.positions_xyz_m[int(np.argmax(build_response(offset_xyz_m))), :2]
    assert float(np.linalg.norm(peak_xy_m - offset_xyz_m[:2])) <= GRID_SPACING_M


def test_the_response_covariance_is_symmetric_and_positive_semi_definite() -> None:
    covariance_m2 = build_covariance(CENTRE_XYZ_M, CORRELATION)
    assert covariance_m2 == pytest.approx(covariance_m2.T)
    assert np.all(np.linalg.eigvalsh(covariance_m2) >= 0.0)


def measure_semi_axis_m(
    highest_frequency_hz: float, background_percentile: float
) -> float:
    covariance_m2 = compute_point_spread_covariance(
        CENTRE_XYZ_M,
        GRID,
        RECEIVERS_XYZ_M,
        RECEIVER_PAIRS,
        SPEED_OF_SOUND_M_PER_S,
        SAMPLE_RATE_HZ,
        MAXIMUM_LAG_S,
        replace(CORRELATION, highest_frequency_hz=highest_frequency_hz),
        STEERED_RESPONSE_POWER,
        None,
        background_percentile,
    )
    return float(compute_point_spread_semi_axes_m(covariance_m2)[0])


def test_raising_the_background_percentile_shrinks_the_response() -> None:
    semi_axes_m = [
        measure_semi_axis_m(3500.0, percentile) for percentile in (0.0, 40.0, 80.0)
    ]
    assert semi_axes_m[0] > semi_axes_m[1] > semi_axes_m[2]


def test_the_response_width_is_set_by_the_percentile_and_not_by_the_band() -> None:
    """The finding that constrains Tier 2, asserted so it cannot quietly change.

    Widening the correlation band fourfold moves the response semi-axis by a
    couple of per cent, while moving the background percentile from zero to
    eighty moves it by a factor of three. The width of this map is a property
    of the array geometry and of the preparation, not of the correlation lobe.
    So the covariance subtracted in the extent stage is only meaningful against
    an observed covariance prepared identically, and the shape factor has to be
    calibrated at the same percentile it will be applied at.
    """
    band_spread_m = abs(
        measure_semi_axis_m(3500.0, 40.0) - measure_semi_axis_m(800.0, 40.0)
    )
    percentile_spread_m = abs(
        measure_semi_axis_m(3500.0, 80.0) - measure_semi_axis_m(3500.0, 0.0)
    )
    assert percentile_spread_m > 10.0 * band_spread_m


def test_the_semi_axes_are_non_negative_and_ordered() -> None:
    semi_axes_m = compute_point_spread_semi_axes_m(
        build_covariance(CENTRE_XYZ_M, CORRELATION)
    )
    assert semi_axes_m[0] >= semi_axes_m[1] >= 0.0


def test_the_covariance_field_carries_one_matrix_per_node() -> None:
    nodes_xyz_m = np.array([[8.0, 8.0, 0.5], [12.0, 12.0, 0.5], [16.0, 14.0, 0.5]])
    positions_xyz_m, covariances_m2 = build_point_spread_covariance_field(
        nodes_xyz_m,
        GRID,
        RECEIVERS_XYZ_M,
        RECEIVER_PAIRS,
        SPEED_OF_SOUND_M_PER_S,
        SAMPLE_RATE_HZ,
        MAXIMUM_LAG_S,
        CORRELATION,
        STEERED_RESPONSE_POWER,
        None,
        BACKGROUND_PERCENTILE,
    )
    assert positions_xyz_m == pytest.approx(nodes_xyz_m)
    assert covariances_m2.shape == (nodes_xyz_m.shape[0], 2, 2)


def test_a_magnitude_that_does_not_match_the_retained_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="whitening band retains"):
        build_point_spread_correlation_curves(
            CENTRE_XYZ_M,
            RECEIVERS_XYZ_M,
            RECEIVER_PAIRS,
            SPEED_OF_SOUND_M_PER_S,
            SAMPLE_RATE_HZ,
            MAXIMUM_LAG_S,
            CORRELATION,
            np.ones(3),
        )
