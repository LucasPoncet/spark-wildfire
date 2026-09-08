import numpy as np
import pytest

from src.config.simulation_configuration import TriangulationConfiguration
from src.spark.inverse.inverse_variance_fusion import fuse_inverse_variance
from src.spark.inverse.level_ratio_triangulation import (
    compute_position_covariance,
    compute_ranges_from_level_ratio,
    is_near_perpendicular_bisector,
    triangulate_from_level_ratio,
    triangulate_position_xy,
)
from src.utils.array_types import Float64Array

NEGLIGIBLE_VARIANCE: float = 1e-6


def make_observables(
    true_position_xy_m: Float64Array,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
) -> tuple[float, float]:
    range_1_m = float(np.linalg.norm(true_position_xy_m - receiver_1_xy_m))
    range_2_m = float(np.linalg.norm(true_position_xy_m - receiver_2_xy_m))
    return 20.0 * np.log10(range_2_m / range_1_m), range_2_m - range_1_m


@pytest.mark.parametrize(
    "true_position_xy_m",
    [
        np.array([25.0, 70.0]),
        np.array([75.0, 35.0]),
        np.array([40.0, 20.0]),
        np.array([60.0, 85.0]),
        np.array([10.0, 10.0]),
    ],
)
def test_round_trip_recovers_the_true_position(
    true_position_xy_m: Float64Array,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    triangulation: TriangulationConfiguration,
) -> None:
    level_difference_db, path_difference_m = make_observables(
        true_position_xy_m, receiver_1_xy_m, receiver_2_xy_m
    )
    position_xy_m = triangulate_from_level_ratio(
        level_difference_db,
        path_difference_m,
        NEGLIGIBLE_VARIANCE,
        NEGLIGIBLE_VARIANCE,
        receiver_1_xy_m,
        receiver_2_xy_m,
        triangulation,
    ).position_xy_m
    assert np.allclose(position_xy_m, true_position_xy_m, atol=1e-6)


def test_round_trip_recovers_the_true_ranges(
    receiver_1_xy_m: Float64Array, receiver_2_xy_m: Float64Array
) -> None:
    true_position_xy_m = np.array([25.0, 70.0])
    level_difference_db, path_difference_m = make_observables(
        true_position_xy_m, receiver_1_xy_m, receiver_2_xy_m
    )
    range_1_m, range_2_m = compute_ranges_from_level_ratio(
        level_difference_db, path_difference_m
    )
    assert range_1_m == pytest.approx(
        np.linalg.norm(true_position_xy_m - receiver_1_xy_m)
    )
    assert range_2_m == pytest.approx(
        np.linalg.norm(true_position_xy_m - receiver_2_xy_m)
    )


def test_negative_domain_side_mirrors_across_the_baseline(
    receiver_1_xy_m: Float64Array, receiver_2_xy_m: Float64Array
) -> None:
    true_position_xy_m = np.array([25.0, 70.0])
    level_difference_db, path_difference_m = make_observables(
        true_position_xy_m, receiver_1_xy_m, receiver_2_xy_m
    )
    range_1_m, range_2_m = compute_ranges_from_level_ratio(
        level_difference_db, path_difference_m
    )
    mirrored = triangulate_position_xy(
        range_1_m, range_2_m, receiver_1_xy_m, receiver_2_xy_m, -1.0
    )
    assert mirrored[0] == pytest.approx(true_position_xy_m[0])
    assert mirrored[1] == pytest.approx(-true_position_xy_m[1])


def test_perpendicular_bisector_is_flagged_as_near_singular(
    triangulation: TriangulationConfiguration,
) -> None:
    tolerance = triangulation.near_singular_tolerance
    assert is_near_perpendicular_bisector(0.0, tolerance)
    assert is_near_perpendicular_bisector(0.2, tolerance)
    assert not is_near_perpendicular_bisector(4.0, tolerance)


def test_range_is_undetermined_on_the_perpendicular_bisector(
    receiver_1_xy_m: Float64Array, receiver_2_xy_m: Float64Array
) -> None:
    midpoint = 0.5 * (receiver_1_xy_m + receiver_2_xy_m)
    for offset_m in [10.0, 50.0, 90.0]:
        true_position_xy_m = midpoint + np.array([0.0, offset_m])
        level_difference_db, path_difference_m = make_observables(
            true_position_xy_m, receiver_1_xy_m, receiver_2_xy_m
        )
        assert level_difference_db == pytest.approx(0.0, abs=1e-12)
        assert path_difference_m == pytest.approx(0.0, abs=1e-12)


def test_covariance_is_symmetric_and_positive_semi_definite(
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    triangulation: TriangulationConfiguration,
) -> None:
    level_difference_db, path_difference_m = make_observables(
        np.array([25.0, 70.0]), receiver_1_xy_m, receiver_2_xy_m
    )
    covariance = compute_position_covariance(
        level_difference_db,
        path_difference_m,
        0.01,
        0.04,
        receiver_1_xy_m,
        receiver_2_xy_m,
        triangulation,
    )
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-9)


def test_covariance_grows_with_input_uncertainty(
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    triangulation: TriangulationConfiguration,
) -> None:
    level_difference_db, path_difference_m = make_observables(
        np.array([25.0, 70.0]), receiver_1_xy_m, receiver_2_xy_m
    )
    small = compute_position_covariance(
        level_difference_db,
        path_difference_m,
        0.001,
        0.001,
        receiver_1_xy_m,
        receiver_2_xy_m,
        triangulation,
    )
    large = compute_position_covariance(
        level_difference_db,
        path_difference_m,
        0.1,
        0.1,
        receiver_1_xy_m,
        receiver_2_xy_m,
        triangulation,
    )
    assert np.trace(large) > np.trace(small)


def test_covariance_grows_as_the_source_approaches_the_bisector(
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    triangulation: TriangulationConfiguration,
) -> None:
    midpoint = 0.5 * (receiver_1_xy_m + receiver_2_xy_m)
    traces = []
    for lateral_offset_m in [20.0, 10.0, 4.0, 1.0]:
        true_position_xy_m = midpoint + np.array([lateral_offset_m, 60.0])
        level_difference_db, path_difference_m = make_observables(
            true_position_xy_m, receiver_1_xy_m, receiver_2_xy_m
        )
        traces.append(
            float(
                np.trace(
                    compute_position_covariance(
                        level_difference_db,
                        path_difference_m,
                        0.01,
                        0.01,
                        receiver_1_xy_m,
                        receiver_2_xy_m,
                        triangulation,
                    )
                )
            )
        )
    assert traces == sorted(traces)


def test_identical_receiver_positions_are_rejected(
    receiver_1_xy_m: Float64Array,
) -> None:
    with pytest.raises(ValueError):
        triangulate_position_xy(10.0, 10.0, receiver_1_xy_m, receiver_1_xy_m, 1.0)


def test_inverse_variance_fusion_favours_the_precise_estimate() -> None:
    fused = fuse_inverse_variance(np.array([1.0, 2.0]), np.array([1e-4, 1.0]))
    assert fused.value == pytest.approx(1.0, abs=1e-3)
    assert fused.variance < 1e-4


def test_inverse_variance_fusion_reports_consistent_bands_as_unit_chi_square() -> None:
    variances = np.full(200, 0.25)
    values = np.random.default_rng(0).normal(loc=3.0, scale=0.5, size=200)
    fused = fuse_inverse_variance(values, variances)
    assert fused.reduced_chi_square == pytest.approx(1.0, abs=0.25)


def test_inverse_variance_fusion_rejects_non_positive_variance() -> None:
    with pytest.raises(ValueError):
        fuse_inverse_variance(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
