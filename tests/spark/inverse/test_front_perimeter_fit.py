"""Fitting an elliptical arc to a map, and reading distances off it.

The ray-ellipse solve is checked on shapes whose answers are known in closed
form, because it is the part that is silently wrong rather than visibly broken:
an arc placed at the right distance but the wrong angular support still looks
plausible on a figure.

The angular sector is what makes this an arc rather than the plan's closed
ellipse. A direction outside it carries no distance, so a front covering a
quarter of the circle reports a quarter rather than being extended round.
"""

import numpy as np
import pytest

from src.spark.inverse.front_perimeter_fit import (
    PerimeterParameters,
    compute_front_distance_by_angle,
    fit_front_perimeter,
    render_perimeter_density,
)
from src.spark.inverse.map_deconvolution import (
    PointSpreadOperator,
    apply_operator,
    build_point_spread_operator,
)

GRID_SIZE: int = 120
GRID_SPACING_M: float = 0.5
GRID_SHAPE: tuple[int, int] = (GRID_SIZE, GRID_SIZE)
DOMAIN_EXTENT_M: float = GRID_SIZE * GRID_SPACING_M
CENTRE_XY_M = np.array([30.0, 30.0])
RESPONSE_WIDTH_CELLS: float = 2.5
FULL_CIRCLE_RAD: float = float(np.pi)
ANGLES_RAD = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)


def build_grid() -> np.ndarray:
    axis_m = np.arange(0.5 * GRID_SPACING_M, DOMAIN_EXTENT_M, GRID_SPACING_M)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    return np.stack(
        (grid_x_m.ravel(), grid_y_m.ravel(), np.full(grid_x_m.size, 0.5)), axis=1
    )


def build_operator() -> PointSpreadOperator:
    index = np.arange(GRID_SIZE, dtype=np.float64) - (GRID_SIZE - 1) / 2.0
    offset_x, offset_y = np.meshgrid(index, index)
    response = np.exp(
        -0.5 * (offset_x**2 + offset_y**2) / RESPONSE_WIDTH_CELLS**2
    ).ravel()
    return build_point_spread_operator(response, GRID_SHAPE, GRID_SPACING_M)


def build_circle(radius_m: float, half_width_rad: float) -> PerimeterParameters:
    return PerimeterParameters(
        centre_xy_m=CENTRE_XY_M,
        semi_axis_major_m=radius_m,
        semi_axis_minor_m=radius_m,
        orientation_rad=0.0,
        sector_centre_rad=0.0,
        sector_half_width_rad=half_width_rad,
    )


def test_a_circle_about_the_origin_is_its_own_radius_in_every_direction() -> None:
    distances_m = compute_front_distance_by_angle(
        build_circle(8.0, FULL_CIRCLE_RAD), CENTRE_XY_M, ANGLES_RAD
    )
    assert np.all(np.isfinite(distances_m))
    assert distances_m == pytest.approx(np.full(ANGLES_RAD.size, 8.0), abs=1e-6)


def test_an_ellipse_reaches_its_major_axis_along_its_orientation() -> None:
    parameters = PerimeterParameters(
        centre_xy_m=CENTRE_XY_M,
        semi_axis_major_m=10.0,
        semi_axis_minor_m=4.0,
        orientation_rad=0.0,
        sector_centre_rad=0.0,
        sector_half_width_rad=FULL_CIRCLE_RAD,
    )
    distances_m = compute_front_distance_by_angle(
        parameters, CENTRE_XY_M, np.array([0.0, 0.5 * np.pi])
    )
    assert float(distances_m[0]) == pytest.approx(10.0, abs=1e-6)
    assert float(distances_m[1]) == pytest.approx(4.0, abs=1e-6)


def test_a_rotated_ellipse_carries_its_major_axis_round_with_it() -> None:
    parameters = PerimeterParameters(
        centre_xy_m=CENTRE_XY_M,
        semi_axis_major_m=10.0,
        semi_axis_minor_m=4.0,
        orientation_rad=0.5 * np.pi,
        sector_centre_rad=0.0,
        sector_half_width_rad=FULL_CIRCLE_RAD,
    )
    distances_m = compute_front_distance_by_angle(
        parameters, CENTRE_XY_M, np.array([0.0, 0.5 * np.pi])
    )
    assert float(distances_m[0]) == pytest.approx(4.0, abs=1e-6)
    assert float(distances_m[1]) == pytest.approx(10.0, abs=1e-6)


def test_an_offset_origin_sees_a_longer_and_a_shorter_side() -> None:
    offset_origin_xy_m = CENTRE_XY_M - np.array([3.0, 0.0])
    distances_m = compute_front_distance_by_angle(
        build_circle(8.0, FULL_CIRCLE_RAD),
        offset_origin_xy_m,
        np.array([0.0, np.pi]),
    )
    assert float(distances_m[0]) == pytest.approx(11.0, abs=1e-6)
    assert float(distances_m[1]) == pytest.approx(5.0, abs=1e-6)


def test_directions_outside_the_sector_carry_no_distance() -> None:
    """What separates an arc from the plan's closed ellipse."""
    half_width_rad = np.deg2rad(45.0)
    distances_m = compute_front_distance_by_angle(
        build_circle(8.0, half_width_rad), CENTRE_XY_M, ANGLES_RAD
    )
    covered = float(np.mean(np.isfinite(distances_m)))
    assert covered == pytest.approx(2.0 * half_width_rad / (2.0 * np.pi), abs=0.03)


def test_a_full_sector_covers_every_direction() -> None:
    distances_m = compute_front_distance_by_angle(
        build_circle(8.0, FULL_CIRCLE_RAD), CENTRE_XY_M, ANGLES_RAD
    )
    assert float(np.mean(np.isfinite(distances_m))) == pytest.approx(1.0)


def test_a_rendered_arc_carries_unit_mass_on_the_grid() -> None:
    density = render_perimeter_density(
        build_circle(8.0, np.deg2rad(45.0)), build_grid(), CENTRE_XY_M, GRID_SHAPE, 200
    )
    assert float(np.sum(density)) == pytest.approx(1.0)
    assert density.shape == GRID_SHAPE


def test_a_rendered_arc_sits_where_its_sector_points() -> None:
    density = render_perimeter_density(
        build_circle(8.0, np.deg2rad(30.0)), build_grid(), CENTRE_XY_M, GRID_SHAPE, 200
    )
    rows, columns = np.nonzero(density)
    axis_m = np.arange(0.5 * GRID_SPACING_M, DOMAIN_EXTENT_M, GRID_SPACING_M)
    mean_x_m = float(np.mean(axis_m[columns]))
    mean_y_m = float(np.mean(axis_m[rows]))
    assert mean_x_m > CENTRE_XY_M[0] + 6.0
    assert mean_y_m == pytest.approx(CENTRE_XY_M[1], abs=1.0)


def test_a_fit_recovers_the_arc_it_was_given() -> None:
    operator = build_operator()
    positions_xyz_m = build_grid()
    truth = build_circle(8.0, np.deg2rad(50.0))
    observed = apply_operator(
        operator,
        render_perimeter_density(truth, positions_xyz_m, CENTRE_XY_M, GRID_SHAPE, 240),
    )
    seed = PerimeterParameters(
        centre_xy_m=CENTRE_XY_M + np.array([1.0, 1.0]),
        semi_axis_major_m=6.0,
        semi_axis_minor_m=6.0,
        orientation_rad=0.0,
        sector_centre_rad=np.deg2rad(10.0),
        sector_half_width_rad=np.deg2rad(60.0),
    )
    fit = fit_front_perimeter(
        observed.ravel(), positions_xyz_m, operator, CENTRE_XY_M, seed, 300, 240
    )
    recovered_m = compute_front_distance_by_angle(
        fit.parameters, CENTRE_XY_M, np.array([0.0])
    )
    assert fit.correlation > 0.9
    assert float(recovered_m[0]) == pytest.approx(8.0, abs=1.5)


def test_a_fit_reports_the_correlation_it_achieved() -> None:
    operator = build_operator()
    positions_xyz_m = build_grid()
    observed = apply_operator(
        operator,
        render_perimeter_density(
            build_circle(8.0, np.deg2rad(50.0)),
            positions_xyz_m,
            CENTRE_XY_M,
            GRID_SHAPE,
            240,
        ),
    )
    fit = fit_front_perimeter(
        observed.ravel(),
        positions_xyz_m,
        operator,
        CENTRE_XY_M,
        build_circle(8.0, np.deg2rad(50.0)),
        120,
        240,
    )
    assert 0.0 <= fit.correlation <= 1.0
    assert fit.residual >= 0.0
