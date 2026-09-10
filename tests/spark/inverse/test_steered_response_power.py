import numpy as np
import pytest

from src.config.multi_source_localization_configuration import (
    HARMONIC_MEAN_COMBINATOR,
    MAXIMUM_POOLING,
    MEAN_POOLING,
    POINT_POOLING,
    PRODUCT_COMBINATOR,
    SUM_COMBINATOR,
    SUM_POOLING,
)
from src.spark.inverse.multilateration import compute_predicted_time_differences_s
from src.spark.inverse.receiver_pair_index import enumerate_receiver_pairs
from src.spark.inverse.steered_response_power import (
    build_candidate_grid_xyz,
    combine_pairwise_maps,
    compute_cell_delay_bounds_s,
    compute_pair_delay_table_s,
    compute_steered_response_power_map,
    extract_map_peak,
    select_retained_cells,
    subdivide_candidate_cells,
)
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
)

SPEED_OF_SOUND_M_PER_S: float = 340.0
DOMAIN_EXTENT_M: float = 100.0
SOURCE_HEIGHT_M: float = 0.5
RECEIVER_HEIGHT_M: float = 1.5
PEAK_WIDTH_SAMPLES: float = 2.0
COARSE_SPACING_M: float = 2.0


def build_receivers_xyz_m() -> np.ndarray:
    bearings_rad = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    return np.stack(
        (
            50.0 + 45.0 * np.cos(bearings_rad),
            50.0 + 45.0 * np.sin(bearings_rad),
            np.full(4, RECEIVER_HEIGHT_M),
        ),
        axis=1,
    )


def build_synthetic_curves(
    source_position_xyz_m: np.ndarray,
    receivers_xyz_m: np.ndarray,
    receiver_pairs: np.ndarray,
    sample_rate_hz: int,
    maximum_absolute_lag_s: float,
) -> list[GeneralizedCrossCorrelationCurve]:
    """One narrow, noiseless peak per pair, exactly at the true delay."""
    true_delays_s = compute_predicted_time_differences_s(
        source_position_xyz_m, receivers_xyz_m, receiver_pairs, SPEED_OF_SOUND_M_PER_S
    )
    maximum_shift = round(maximum_absolute_lag_s * sample_rate_hz)
    lags_s = np.arange(-maximum_shift, maximum_shift + 1, dtype=np.float64) / (
        sample_rate_hz
    )
    peak_width_s = PEAK_WIDTH_SAMPLES / sample_rate_hz
    return [
        GeneralizedCrossCorrelationCurve(
            receiver_pair=(int(pair[0]), int(pair[1])),
            lags_s=lags_s,
            values=np.exp(-0.5 * ((lags_s - float(delay_s)) / peak_width_s) ** 2),
            sample_rate_hz=sample_rate_hz,
            window_count=1,
            effective_bandwidth_hz=5000.0,
        )
        for pair, delay_s in zip(receiver_pairs, true_delays_s, strict=True)
    ]


def test_the_candidate_grid_tiles_the_domain_at_the_requested_spacing() -> None:
    grid = build_candidate_grid_xyz(
        DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, 10.0, SOURCE_HEIGHT_M
    )
    assert grid.positions_xyz_m.shape == (100, 3)
    assert grid.cell_extent_m == 10.0
    assert np.allclose(grid.positions_xyz_m[:, 2], SOURCE_HEIGHT_M)


def test_a_non_positive_grid_spacing_is_rejected() -> None:
    with pytest.raises(ValueError, match="spacing must be positive"):
        build_candidate_grid_xyz(10.0, 10.0, 0.0, SOURCE_HEIGHT_M)


def test_subdivision_multiplies_the_cell_count_and_shrinks_the_cells() -> None:
    grid = build_candidate_grid_xyz(20.0, 20.0, 10.0, SOURCE_HEIGHT_M)
    finer = subdivide_candidate_cells(grid.positions_xyz_m, grid.cell_extent_m, 4)
    assert finer.positions_xyz_m.shape[0] == 16 * grid.positions_xyz_m.shape[0]
    assert finer.cell_extent_m == pytest.approx(2.5)


def test_the_delay_bounds_contain_every_delay_the_cell_can_produce() -> None:
    receivers_xyz_m = build_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    cell_centre_xyz_m = np.array([[40.0, 60.0, SOURCE_HEIGHT_M]])
    cell_extent_m = 4.0
    lower_s, upper_s = compute_cell_delay_bounds_s(
        cell_centre_xyz_m,
        cell_extent_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
    )

    generator = np.random.default_rng(0)
    offsets_m = generator.uniform(
        -0.5 * cell_extent_m, 0.5 * cell_extent_m, size=(500, 2)
    )
    samples_xyz_m = np.column_stack(
        (cell_centre_xyz_m[0, :2] + offsets_m, np.full(500, SOURCE_HEIGHT_M))
    )
    sampled_delays_s = compute_pair_delay_table_s(
        samples_xyz_m, receivers_xyz_m, receiver_pairs, SPEED_OF_SOUND_M_PER_S
    )
    assert np.all(sampled_delays_s >= lower_s - 1e-12)
    assert np.all(sampled_delays_s <= upper_s + 1e-12)


def test_volumetric_pooling_finds_a_source_point_sampling_misses(
    sample_rate_hz: int,
) -> None:
    """The revision-2 correction, made load-bearing.

    A two-metre cell spans hundreds of samples of delay while the correlation
    peak is two samples wide, so evaluating at the cell centre steps over the
    peak. Volumetric pooling reads the same curves over the interval the cell
    actually covers and lands on the right cell; point sampling does not.
    """
    receivers_xyz_m = build_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    true_position_xyz_m = np.array([31.7, 67.3, SOURCE_HEIGHT_M])
    curves = build_synthetic_curves(
        true_position_xyz_m,
        receivers_xyz_m,
        receiver_pairs,
        sample_rate_hz,
        0.35,
    )
    grid = build_candidate_grid_xyz(
        DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, COARSE_SPACING_M, SOURCE_HEIGHT_M
    )
    lower_s, upper_s = compute_cell_delay_bounds_s(
        grid.positions_xyz_m,
        grid.cell_extent_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
    )
    true_cell_index = int(
        np.argmin(
            np.linalg.norm(
                grid.positions_xyz_m[:, :2] - true_position_xyz_m[:2], axis=1
            )
        )
    )

    volumetric_map = compute_steered_response_power_map(
        curves, lower_s, upper_s, MAXIMUM_POOLING, PRODUCT_COMBINATOR
    )
    point_map = compute_steered_response_power_map(
        curves, lower_s, upper_s, POINT_POOLING, PRODUCT_COMBINATOR
    )

    assert int(np.argmax(volumetric_map)) == true_cell_index
    assert int(np.argmax(point_map)) != true_cell_index
    assert point_map[true_cell_index] < 1e-3 * float(np.max(volumetric_map))


def test_every_pooling_rule_and_combinator_peaks_at_the_truth(
    sample_rate_hz: int,
) -> None:
    receivers_xyz_m = build_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    true_position_xyz_m = np.array([31.7, 67.3, SOURCE_HEIGHT_M])
    curves = build_synthetic_curves(
        true_position_xyz_m, receivers_xyz_m, receiver_pairs, sample_rate_hz, 0.35
    )
    grid = build_candidate_grid_xyz(
        DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, COARSE_SPACING_M, SOURCE_HEIGHT_M
    )
    lower_s, upper_s = compute_cell_delay_bounds_s(
        grid.positions_xyz_m,
        grid.cell_extent_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
    )
    cell_diagonal_m = COARSE_SPACING_M * np.sqrt(2.0)
    for pooling in (SUM_POOLING, MEAN_POOLING, MAXIMUM_POOLING):
        for combinator in (
            SUM_COMBINATOR,
            PRODUCT_COMBINATOR,
            HARMONIC_MEAN_COMBINATOR,
        ):
            position_xyz_m, _, _ = extract_map_peak(
                compute_steered_response_power_map(
                    curves, lower_s, upper_s, pooling, combinator
                ),
                grid.positions_xyz_m,
            )
            assert (
                float(np.linalg.norm(position_xyz_m[:2] - true_position_xyz_m[:2]))
                <= cell_diagonal_m
            ), f"{pooling} pooling with the {combinator} combinator missed"


def test_an_unknown_pooling_rule_or_combinator_is_rejected(
    sample_rate_hz: int,
) -> None:
    receivers_xyz_m = build_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    curves = build_synthetic_curves(
        np.array([31.0, 67.0, SOURCE_HEIGHT_M]),
        receivers_xyz_m,
        receiver_pairs,
        sample_rate_hz,
        0.35,
    )
    bounds = np.zeros((1, receiver_pairs.shape[0]))
    with pytest.raises(ValueError, match="unknown pooling rule"):
        compute_steered_response_power_map(
            curves, bounds, bounds, "nonsense", SUM_COMBINATOR
        )
    with pytest.raises(ValueError, match="unknown pairwise combinator"):
        combine_pairwise_maps(np.ones((2, 3)), "nonsense")


def test_mismatched_bound_columns_are_rejected(sample_rate_hz: int) -> None:
    receivers_xyz_m = build_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    curves = build_synthetic_curves(
        np.array([31.0, 67.0, SOURCE_HEIGHT_M]),
        receivers_xyz_m,
        receiver_pairs,
        sample_rate_hz,
        0.35,
    )
    bounds = np.zeros((4, 2))
    with pytest.raises(ValueError, match="one column of delay bounds"):
        compute_steered_response_power_map(
            curves, bounds, bounds, MAXIMUM_POOLING, SUM_COMBINATOR
        )


def test_retaining_a_fraction_keeps_the_strongest_cells() -> None:
    grid = build_candidate_grid_xyz(10.0, 10.0, 1.0, SOURCE_HEIGHT_M)
    power_map = np.arange(grid.positions_xyz_m.shape[0], dtype=np.float64)
    retained = select_retained_cells(power_map, grid.positions_xyz_m, 0.1)
    assert retained.shape[0] == 10
    assert np.allclose(
        np.sort(retained[:, 0]),
        np.sort(grid.positions_xyz_m[-10:, 0]),
    )


def test_an_out_of_range_retained_fraction_is_rejected() -> None:
    grid = build_candidate_grid_xyz(10.0, 10.0, 1.0, SOURCE_HEIGHT_M)
    with pytest.raises(ValueError, match=r"retained_fraction must lie in \(0, 1\]"):
        select_retained_cells(np.zeros(100), grid.positions_xyz_m, 0.0)
