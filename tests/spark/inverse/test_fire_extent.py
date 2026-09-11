"""Deconvolving the array response out of a second moment.

The two cases that matter are the one the plan's gate names and the one it
warns about. A ring wider than the response must come back at its true radius;
a ring narrower than it must come back flagged, with an upper bound that
contains the truth, and never as a number clipped to zero and presented as a
measurement.
"""

import numpy as np
import pytest

from src.spark.inverse.fire_extent import (
    THIN_RING_SHAPE_FACTOR,
    estimate_fire_extent,
    select_resolved,
)

GRID_SPACING_M: float = 0.25
DOMAIN_EXTENT_M: float = 60.0
CENTRE_XY_M = np.array([30.0, 30.0])
NO_BACKGROUND: float = 0.0
RESPONSE_WIDTH_M: float = 1.5


def build_grid() -> np.ndarray:
    axis_m = np.arange(0.5 * GRID_SPACING_M, DOMAIN_EXTENT_M, GRID_SPACING_M)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    return np.stack(
        (grid_x_m.ravel(), grid_y_m.ravel(), np.full(grid_x_m.size, 0.5)), axis=1
    )


def build_blurred_ring(
    positions_xyz_m: np.ndarray, radius_m: float, response_width_m: float
) -> np.ndarray:
    """A ring convolved with an isotropic response, to within a Gaussian."""
    radii_m = np.linalg.norm(positions_xyz_m[:, :2] - CENTRE_XY_M, axis=1)
    return np.asarray(
        np.exp(-0.5 * ((radii_m - radius_m) / response_width_m) ** 2),
        dtype=np.float64,
    )


def response_covariance_m2(width_m: float) -> np.ndarray:
    return np.diag([width_m**2, width_m**2])


def test_a_ring_wider_than_the_response_is_recovered_to_its_radius() -> None:
    positions_xyz_m = build_grid()
    radius_m = 8.0
    extent = estimate_fire_extent(
        10.0,
        build_blurred_ring(positions_xyz_m, radius_m, RESPONSE_WIDTH_M),
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(RESPONSE_WIDTH_M),
        THIN_RING_SHAPE_FACTOR,
    )
    assert extent.is_resolved
    assert extent.semi_axis_major_m == pytest.approx(radius_m, rel=0.05)


@pytest.mark.parametrize("radius_m", [5.0, 8.0, 12.0])
def test_the_recovered_radius_tracks_the_true_one(radius_m: float) -> None:
    positions_xyz_m = build_grid()
    extent = estimate_fire_extent(
        10.0,
        build_blurred_ring(positions_xyz_m, radius_m, RESPONSE_WIDTH_M),
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(RESPONSE_WIDTH_M),
        THIN_RING_SHAPE_FACTOR,
    )
    assert extent.semi_axis_major_m == pytest.approx(radius_m, rel=0.05)


def test_subtracting_the_response_is_what_makes_it_accurate() -> None:
    positions_xyz_m = build_grid()
    radius_m = 6.0
    blurred = build_blurred_ring(positions_xyz_m, radius_m, RESPONSE_WIDTH_M)
    deconvolved = estimate_fire_extent(
        10.0,
        blurred,
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(RESPONSE_WIDTH_M),
        THIN_RING_SHAPE_FACTOR,
    )
    undeconvolved = estimate_fire_extent(
        10.0,
        blurred,
        positions_xyz_m,
        NO_BACKGROUND,
        np.zeros((2, 2)),
        THIN_RING_SHAPE_FACTOR,
    )
    assert abs(deconvolved.semi_axis_major_m - radius_m) < abs(
        undeconvolved.semi_axis_major_m - radius_m
    )


def test_a_source_smaller_than_the_response_is_flagged_not_clipped() -> None:
    positions_xyz_m = build_grid()
    compact = build_blurred_ring(positions_xyz_m, 0.0, 1.0)
    extent = estimate_fire_extent(
        5.0,
        compact,
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(10.0),
        THIN_RING_SHAPE_FACTOR,
    )
    assert not extent.is_resolved
    assert extent.upper_bound_only


def test_an_unresolved_upper_bound_contains_the_truth() -> None:
    positions_xyz_m = build_grid()
    true_radius_m = 1.0
    extent = estimate_fire_extent(
        5.0,
        build_blurred_ring(positions_xyz_m, true_radius_m, 1.0),
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(10.0),
        THIN_RING_SHAPE_FACTOR,
    )
    assert not extent.is_resolved
    assert extent.semi_axis_major_m >= true_radius_m


def test_an_unresolved_frame_never_reports_a_clipped_zero() -> None:
    positions_xyz_m = build_grid()
    extent = estimate_fire_extent(
        5.0,
        build_blurred_ring(positions_xyz_m, 0.5, 1.0),
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(20.0),
        THIN_RING_SHAPE_FACTOR,
    )
    assert extent.semi_axis_major_m > 0.0
    assert extent.observed_semi_axis_major_m > 0.0


def test_an_elongated_source_reports_its_orientation() -> None:
    positions_xyz_m = build_grid()
    offsets_m = positions_xyz_m[:, :2] - CENTRE_XY_M
    elongated = np.exp(
        -0.5 * ((offsets_m[:, 0] / 8.0) ** 2 + (offsets_m[:, 1] / 2.0) ** 2)
    )
    extent = estimate_fire_extent(
        10.0,
        elongated,
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(0.5),
        THIN_RING_SHAPE_FACTOR,
    )
    assert extent.is_resolved
    assert extent.semi_axis_major_m > extent.semi_axis_minor_m
    assert abs(np.rad2deg(extent.orientation_rad)) % 180.0 < 2.0


def test_the_centroid_comes_back_with_the_extent() -> None:
    positions_xyz_m = build_grid()
    extent = estimate_fire_extent(
        10.0,
        build_blurred_ring(positions_xyz_m, 6.0, RESPONSE_WIDTH_M),
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(RESPONSE_WIDTH_M),
        THIN_RING_SHAPE_FACTOR,
    )
    assert extent.centroid_xy_m == pytest.approx(CENTRE_XY_M, abs=0.2)


def test_only_resolved_frames_are_selected() -> None:
    positions_xyz_m = build_grid()
    resolved = estimate_fire_extent(
        10.0,
        build_blurred_ring(positions_xyz_m, 8.0, RESPONSE_WIDTH_M),
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(RESPONSE_WIDTH_M),
        THIN_RING_SHAPE_FACTOR,
    )
    unresolved = estimate_fire_extent(
        5.0,
        build_blurred_ring(positions_xyz_m, 0.5, 1.0),
        positions_xyz_m,
        NO_BACKGROUND,
        response_covariance_m2(20.0),
        THIN_RING_SHAPE_FACTOR,
    )
    assert select_resolved([resolved, unresolved]) == [resolved]


def test_the_thin_ring_shape_factor_is_the_root_of_two() -> None:
    assert float(THIN_RING_SHAPE_FACTOR) == pytest.approx(float(np.sqrt(2.0)))
