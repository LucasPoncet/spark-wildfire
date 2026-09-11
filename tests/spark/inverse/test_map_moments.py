"""Moments of maps whose answers are known in closed form.

The thin-ring case is the one Tier 2 rests on: a ring of radius `R` has second
central moment `R^2 / 2` about each axis, which is what lets a front's radius
be recovered from a map that never shows the ring as a ring. It is asserted
here exactly rather than approximately, because the masses are placed at exact
positions rather than snapped to a grid.

Every fixture carries explicit zero-valued cells alongside its mass. Without
them the map's own minimum is its only value, so the background percentile
subtracts the whole frame and the moments are taken of nothing — which is
correct behaviour on a map with no floor, and not what any real frame looks
like. The zero cells contribute no mass and so move no moment.
"""

import numpy as np
import pytest

from src.spark.inverse.map_moments import (
    compute_directional_skewness,
    compute_directional_third_moment,
    compute_map_moments,
    compute_skewness_over_directions,
)

NO_BACKGROUND: float = 0.0
RING_RADIUS_M: float = 7.0
CENTRE_XY_M = np.array([30.0, 30.0])


def add_zero_floor(
    values: np.ndarray, positions_xyz_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    floor_positions_xyz_m = np.column_stack(
        (
            np.full(values.size, CENTRE_XY_M[0]),
            np.full(values.size, CENTRE_XY_M[1]),
            np.full(values.size, 0.5),
        )
    )
    return (
        np.concatenate((values, np.zeros(values.size))),
        np.concatenate((positions_xyz_m, floor_positions_xyz_m)),
    )


def build_ring(point_count: int, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    angles_rad = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    positions_xyz_m = np.stack(
        (
            CENTRE_XY_M[0] + radius_m * np.cos(angles_rad),
            CENTRE_XY_M[1] + radius_m * np.sin(angles_rad),
            np.full(point_count, 0.5),
        ),
        axis=1,
    )
    return add_zero_floor(np.ones(point_count), positions_xyz_m)


def build_gaussian(
    standard_deviation_x_m: float, standard_deviation_y_m: float, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    half_width_m = 8.0 * max(standard_deviation_x_m, standard_deviation_y_m)
    axis_m = np.arange(-half_width_m, half_width_m + spacing_m, spacing_m)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    values = np.exp(
        -0.5
        * (
            (grid_x_m / standard_deviation_x_m) ** 2
            + (grid_y_m / standard_deviation_y_m) ** 2
        )
    ).ravel()
    positions_xyz_m = np.stack(
        (
            grid_x_m.ravel() + CENTRE_XY_M[0],
            grid_y_m.ravel() + CENTRE_XY_M[1],
            np.full(grid_x_m.size, 0.5),
        ),
        axis=1,
    )
    return values, positions_xyz_m


def test_a_ring_puts_its_centroid_at_its_centre() -> None:
    values, positions_xyz_m = build_ring(64, RING_RADIUS_M)
    moments = compute_map_moments(values, positions_xyz_m, NO_BACKGROUND)
    assert moments.centroid_xy_m == pytest.approx(CENTRE_XY_M, abs=1e-9)


def test_a_ring_has_half_its_squared_radius_as_variance_on_each_axis() -> None:
    values, positions_xyz_m = build_ring(64, RING_RADIUS_M)
    moments = compute_map_moments(values, positions_xyz_m, NO_BACKGROUND)
    assert moments.principal_variances_m2 == pytest.approx(
        [RING_RADIUS_M**2 / 2.0, RING_RADIUS_M**2 / 2.0], abs=1e-9
    )


def test_the_ring_radius_is_recovered_by_the_root_two_shape_factor() -> None:
    values, positions_xyz_m = build_ring(64, RING_RADIUS_M)
    moments = compute_map_moments(values, positions_xyz_m, NO_BACKGROUND)
    recovered_radius_m = np.sqrt(2.0) * np.sqrt(moments.principal_variances_m2[0])
    assert recovered_radius_m == pytest.approx(RING_RADIUS_M, rel=1e-9)


def test_a_ring_is_isotropic_so_its_variances_are_equal() -> None:
    values, positions_xyz_m = build_ring(32, RING_RADIUS_M)
    moments = compute_map_moments(values, positions_xyz_m, NO_BACKGROUND)
    assert moments.principal_variances_m2[0] == pytest.approx(
        moments.principal_variances_m2[1], abs=1e-9
    )


def test_four_point_masses_give_their_exact_covariance() -> None:
    offsets_m = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 5.0], [0.0, -5.0]])
    positions_xyz_m = np.column_stack(
        (offsets_m + CENTRE_XY_M, np.full(offsets_m.shape[0], 0.5))
    )
    values, positions_xyz_m = add_zero_floor(np.ones(4), positions_xyz_m)
    moments = compute_map_moments(values, positions_xyz_m, NO_BACKGROUND)
    assert moments.covariance_xy_m2 == pytest.approx(
        np.diag([9.0 / 2.0, 25.0 / 2.0]), abs=1e-9
    )
    assert moments.principal_variances_m2 == pytest.approx([12.5, 4.5], abs=1e-9)


def test_a_gaussian_recovers_its_own_standard_deviations() -> None:
    values, positions_xyz_m = build_gaussian(4.0, 2.0, 0.05)
    moments = compute_map_moments(values, positions_xyz_m, NO_BACKGROUND)
    assert moments.centroid_xy_m == pytest.approx(CENTRE_XY_M, abs=1e-6)
    assert np.sqrt(moments.principal_variances_m2) == pytest.approx(
        [4.0, 2.0], rel=1e-6
    )


def test_the_major_axis_of_an_elongated_gaussian_points_along_its_long_side() -> None:
    values, positions_xyz_m = build_gaussian(4.0, 1.0, 0.1)
    moments = compute_map_moments(values, positions_xyz_m, NO_BACKGROUND)
    assert abs(float(moments.principal_axes[0, 0])) == pytest.approx(1.0, abs=1e-6)


def test_a_symmetric_map_has_no_skewness_in_any_direction() -> None:
    values, positions_xyz_m = build_ring(64, RING_RADIUS_M)
    _, skewness = compute_skewness_over_directions(
        values, positions_xyz_m, CENTRE_XY_M, 36, NO_BACKGROUND
    )
    assert np.max(np.abs(skewness)) == pytest.approx(0.0, abs=1e-9)


def test_a_brighter_head_skews_the_mass_towards_it() -> None:
    values, positions_xyz_m = build_ring(64, RING_RADIUS_M)
    is_head = positions_xyz_m[:, 0] > CENTRE_XY_M[0]
    values = np.where(is_head, 2.0 * values, values)
    third_moment_m3 = compute_directional_third_moment(
        values, positions_xyz_m, CENTRE_XY_M, np.array([1.0, 0.0]), NO_BACKGROUND
    )
    assert third_moment_m3 > 0.0


def test_the_skewness_sweep_peaks_along_the_brighter_side() -> None:
    values, positions_xyz_m = build_ring(720, RING_RADIUS_M)
    bearing_rad = np.deg2rad(35.0)
    direction = np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
    is_head = (positions_xyz_m[:, :2] - CENTRE_XY_M) @ direction > 0.0
    values = np.where(is_head, 2.0 * values, values)
    angles_rad, skewness = compute_skewness_over_directions(
        values, positions_xyz_m, CENTRE_XY_M, 720, NO_BACKGROUND
    )
    recovered_rad = float(angles_rad[int(np.argmax(skewness))])
    assert np.rad2deg(abs(recovered_rad - bearing_rad)) < 1.0


def test_skewness_is_the_third_moment_scaled_by_the_second() -> None:
    values, positions_xyz_m = build_ring(128, RING_RADIUS_M)
    values = np.where(positions_xyz_m[:, 1] > CENTRE_XY_M[1], 3.0 * values, values)
    direction = np.array([0.0, 1.0])
    moments = compute_map_moments(values, positions_xyz_m, NO_BACKGROUND)
    third_moment_m3 = compute_directional_third_moment(
        values, positions_xyz_m, moments.centroid_xy_m, direction, NO_BACKGROUND
    )
    skewness = compute_directional_skewness(
        values, positions_xyz_m, moments.centroid_xy_m, direction, NO_BACKGROUND
    )
    variance_m2 = float(direction @ moments.covariance_xy_m2 @ direction)
    assert skewness == pytest.approx(third_moment_m3 / variance_m2**1.5, rel=1e-9)


def test_a_zero_length_direction_is_rejected() -> None:
    values, positions_xyz_m = build_ring(16, RING_RADIUS_M)
    with pytest.raises(ValueError, match="non-zero length"):
        compute_directional_third_moment(
            values, positions_xyz_m, CENTRE_XY_M, np.zeros(2), NO_BACKGROUND
        )


def test_a_sweep_of_fewer_than_three_directions_is_rejected() -> None:
    values, positions_xyz_m = build_ring(16, RING_RADIUS_M)
    with pytest.raises(ValueError, match="at least three directions"):
        compute_skewness_over_directions(
            values, positions_xyz_m, CENTRE_XY_M, 2, NO_BACKGROUND
        )


def test_an_empty_frame_reports_no_mass_rather_than_dividing_by_zero() -> None:
    _, positions_xyz_m = build_ring(16, RING_RADIUS_M)
    moments = compute_map_moments(np.zeros(16), positions_xyz_m, NO_BACKGROUND)
    assert moments.total_mass == 0.0
    assert moments.covariance_xy_m2 == pytest.approx(np.zeros((2, 2)))
