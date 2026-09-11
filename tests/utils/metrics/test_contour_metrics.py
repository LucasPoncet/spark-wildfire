"""The three contour metrics, on shapes whose answers are known by hand.

The overlap is the one worth checking carefully. It replaces the plan's
enclosed-area ratio, which is undefined for an open arc, so it has to be shown
to do two things: reduce to the area ratio when both contours close, and charge
something for coverage one has and the other does not.
"""

import numpy as np
import pytest

from src.utils.metrics.contour_metrics import (
    compute_coverage_agreement,
    compute_hausdorff_distance,
    compute_mean_radial_error,
    compute_sector_overlap,
)

ANGLE_COUNT: int = 72


def build_ring(radius_m: float) -> np.ndarray:
    return np.full(ANGLE_COUNT, radius_m, dtype=np.float64)


def build_arc(radius_m: float, covered_count: int) -> np.ndarray:
    distances_m = np.full(ANGLE_COUNT, np.nan, dtype=np.float64)
    distances_m[:covered_count] = radius_m
    return distances_m


def test_identical_contours_have_no_hausdorff_distance() -> None:
    square_xy_m = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert compute_hausdorff_distance(square_xy_m, square_xy_m) == pytest.approx(0.0)


def test_a_displaced_contour_reports_its_displacement() -> None:
    """A unit square pushed three metres along x sits three metres away.

    Every corner's nearest counterpart is the one directly across the gap, so
    the worst of those is the gap itself rather than the corner-to-far-corner
    diagonal.
    """
    square_xy_m = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert compute_hausdorff_distance(
        square_xy_m, square_xy_m + np.array([3.0, 0.0])
    ) == pytest.approx(3.0)


def test_hausdorff_catches_one_stray_point() -> None:
    base_xy_m = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with_spur_xy_m = np.vstack([base_xy_m, np.array([[0.5, 9.0]])])
    assert compute_hausdorff_distance(with_spur_xy_m, base_xy_m) > 7.9


def test_an_empty_contour_is_infinitely_far_off() -> None:
    assert compute_hausdorff_distance(
        np.empty((0, 2)), np.array([[0.0, 0.0]])
    ) == float("inf")


def test_a_constant_offset_in_radius_is_the_mean_radial_error() -> None:
    assert compute_mean_radial_error(build_ring(9.0), build_ring(7.0)) == pytest.approx(
        2.0
    )


def test_the_radial_error_ignores_directions_only_one_side_covers() -> None:
    estimated = build_arc(7.5, 36)
    true_values = build_arc(7.0, 18)
    assert compute_mean_radial_error(estimated, true_values) == pytest.approx(0.5)


def test_contours_that_share_no_direction_have_no_radial_error() -> None:
    estimated = np.full(ANGLE_COUNT, np.nan)
    estimated[:10] = 5.0
    true_values = np.full(ANGLE_COUNT, np.nan)
    true_values[40:50] = 5.0
    assert np.isnan(compute_mean_radial_error(estimated, true_values))


def test_mismatched_series_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one true distance"):
        compute_mean_radial_error(np.zeros(3), np.zeros(2))


def test_identical_rings_overlap_completely() -> None:
    assert compute_sector_overlap(build_ring(8.0), build_ring(8.0)) == pytest.approx(
        1.0
    )


def test_the_overlap_of_two_full_rings_is_the_ratio_of_their_areas() -> None:
    """Where both contours close, it reduces to the plan's enclosed-area form."""
    assert compute_sector_overlap(build_ring(6.0), build_ring(8.0)) == pytest.approx(
        (6.0**2) / (8.0**2)
    )


def test_coverage_one_side_lacks_costs_overlap() -> None:
    half = build_arc(8.0, ANGLE_COUNT // 2)
    assert compute_sector_overlap(half, build_ring(8.0)) == pytest.approx(0.5)


def test_two_contours_claiming_nothing_overlap_by_nothing() -> None:
    empty = np.full(ANGLE_COUNT, np.nan)
    assert compute_sector_overlap(empty, empty) == pytest.approx(0.0)


def test_matching_coverage_is_full_agreement() -> None:
    arc = build_arc(8.0, 18)
    assert compute_coverage_agreement(arc, arc) == pytest.approx(1.0)


def test_opposite_coverage_is_no_agreement() -> None:
    first = build_arc(8.0, ANGLE_COUNT // 2)
    second = np.full(ANGLE_COUNT, 8.0)
    second[: ANGLE_COUNT // 2] = np.nan
    assert compute_coverage_agreement(first, second) == pytest.approx(0.0)


def test_a_quarter_arc_against_a_full_ring_agrees_on_a_quarter() -> None:
    assert compute_coverage_agreement(
        build_arc(8.0, ANGLE_COUNT // 4), build_ring(8.0)
    ) == pytest.approx(0.25)
