"""Reading a front's outer edge off a density, direction by direction.

The uncovered-direction case is the one that matters. A radiating fire front is
an open arc, so an extractor that returned a distance in every direction would
be inventing three quarters of a curve, and every metric taken against it would
be measuring the invention. What is asserted here is that a direction with no
front reports none.
"""

import numpy as np
import pytest

from src.spark.inverse.front_contour import (
    compute_angular_coverage_fraction,
    compute_contour_distance_by_angle,
    extract_front_contour,
)

GRID_SPACING_M: float = 0.25
DOMAIN_EXTENT_M: float = 40.0
CENTRE_XY_M = np.array([20.0, 20.0])
LEVEL_FRACTION: float = 0.5
MAXIMUM_RADIUS_M: float = 18.0
RADIUS_STEP_M: float = 0.1
ANGLES_RAD = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)


def build_grid() -> np.ndarray:
    axis_m = np.arange(0.5 * GRID_SPACING_M, DOMAIN_EXTENT_M, GRID_SPACING_M)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    return np.stack(
        (grid_x_m.ravel(), grid_y_m.ravel(), np.full(grid_x_m.size, 0.5)), axis=1
    )


def build_disc(positions_xyz_m: np.ndarray, radius_m: float) -> np.ndarray:
    radii_m = np.linalg.norm(positions_xyz_m[:, :2] - CENTRE_XY_M, axis=1)
    return np.asarray(
        np.exp(-0.5 * (radii_m / (0.45 * radius_m)) ** 2), dtype=np.float64
    )


def build_arc(
    positions_xyz_m: np.ndarray, radius_m: float, half_width_rad: float
) -> np.ndarray:
    offsets_m = positions_xyz_m[:, :2] - CENTRE_XY_M
    radii_m = np.linalg.norm(offsets_m, axis=1)
    angles_rad = np.arctan2(offsets_m[:, 1], offsets_m[:, 0])
    ring = np.exp(-0.5 * ((radii_m - radius_m) / 0.8) ** 2)
    inside = (
        np.abs(np.arctan2(np.sin(angles_rad), np.cos(angles_rad))) <= half_width_rad
    )
    return np.asarray(np.where(inside, ring, 0.0), dtype=np.float64)


def extract(density: np.ndarray) -> np.ndarray:
    return compute_contour_distance_by_angle(
        density,
        build_grid(),
        CENTRE_XY_M,
        ANGLES_RAD,
        LEVEL_FRACTION,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )


@pytest.mark.parametrize("radius_m", [5.0, 9.0])
def test_a_symmetric_disc_gives_one_radius_in_every_direction(radius_m: float) -> None:
    distances_m = extract(build_disc(build_grid(), radius_m))
    assert np.all(np.isfinite(distances_m))
    assert float(np.std(distances_m)) < 0.2


def test_a_closed_shape_is_covered_everywhere() -> None:
    assert compute_angular_coverage_fraction(
        extract(build_disc(build_grid(), 7.0))
    ) == pytest.approx(1.0)


def test_an_arc_reports_no_front_where_it_has_none() -> None:
    """The case a closed-contour extractor would have to invent."""
    half_width_rad = np.deg2rad(45.0)
    distances_m = extract(build_arc(build_grid(), 8.0, half_width_rad))
    covered = compute_angular_coverage_fraction(distances_m)
    assert covered == pytest.approx(2.0 * half_width_rad / (2.0 * np.pi), abs=0.06)
    assert not np.all(np.isfinite(distances_m))


def test_the_covered_directions_are_the_ones_the_arc_spans() -> None:
    distances_m = extract(build_arc(build_grid(), 8.0, np.deg2rad(45.0)))
    covered_angles_rad = ANGLES_RAD[np.isfinite(distances_m)]
    wrapped_deg = np.rad2deg(
        np.arctan2(np.sin(covered_angles_rad), np.cos(covered_angles_rad))
    )
    assert float(np.max(np.abs(wrapped_deg))) <= 50.0


def test_an_arc_is_found_at_its_outer_edge_not_its_centre() -> None:
    """What "outer edge" means for a front of finite width.

    The arc is a Gaussian ring of width 0.8 m about 8 m, and the extractor
    takes the outermost crossing of half its peak. That crossing sits at
    `radius + width * sqrt(2 ln 2)`, about 0.94 m beyond the ring's centre, and
    it is the leading edge — which is the quantity a front position wants.
    """
    radius_m = 8.0
    ring_width_m = 0.8
    distances_m = extract(build_arc(build_grid(), radius_m, np.deg2rad(45.0)))
    found = distances_m[np.isfinite(distances_m)]
    outer_edge_m = radius_m + ring_width_m * np.sqrt(2.0 * np.log(2.0))
    assert float(np.median(found)) == pytest.approx(outer_edge_m, abs=0.3)
    assert float(np.median(found)) > radius_m


def test_an_empty_density_covers_nothing() -> None:
    positions_xyz_m = build_grid()
    distances_m = extract(np.zeros(positions_xyz_m.shape[0]))
    assert not np.any(np.isfinite(distances_m))
    assert compute_angular_coverage_fraction(distances_m) == pytest.approx(0.0)


def test_the_contour_carries_only_the_covered_directions() -> None:
    positions_xyz_m = build_grid()
    density = build_arc(positions_xyz_m, 8.0, np.deg2rad(45.0))
    distances_m = extract(density)
    contour_xy_m = extract_front_contour(
        density,
        positions_xyz_m,
        CENTRE_XY_M,
        ANGLES_RAD,
        LEVEL_FRACTION,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    assert contour_xy_m.shape[0] == int(np.count_nonzero(np.isfinite(distances_m)))
    assert contour_xy_m.shape[1] == 2


def test_an_empty_density_yields_an_empty_contour() -> None:
    positions_xyz_m = build_grid()
    contour_xy_m = extract_front_contour(
        np.zeros(positions_xyz_m.shape[0]),
        positions_xyz_m,
        CENTRE_XY_M,
        ANGLES_RAD,
        LEVEL_FRACTION,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    assert contour_xy_m.shape == (0, 2)


def test_a_higher_level_finds_a_tighter_contour() -> None:
    positions_xyz_m = build_grid()
    density = build_disc(positions_xyz_m, 9.0)
    low = compute_contour_distance_by_angle(
        density,
        positions_xyz_m,
        CENTRE_XY_M,
        ANGLES_RAD,
        0.3,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    high = compute_contour_distance_by_angle(
        density,
        positions_xyz_m,
        CENTRE_XY_M,
        ANGLES_RAD,
        0.8,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    assert float(np.nanmedian(high)) < float(np.nanmedian(low))


def test_a_level_outside_the_unit_interval_is_rejected() -> None:
    positions_xyz_m = build_grid()
    with pytest.raises(ValueError, match="level fraction"):
        compute_contour_distance_by_angle(
            build_disc(positions_xyz_m, 7.0),
            positions_xyz_m,
            CENTRE_XY_M,
            ANGLES_RAD,
            1.5,
            MAXIMUM_RADIUS_M,
            RADIUS_STEP_M,
        )


def test_a_non_positive_step_is_rejected() -> None:
    positions_xyz_m = build_grid()
    with pytest.raises(ValueError, match="must both be positive"):
        compute_contour_distance_by_angle(
            build_disc(positions_xyz_m, 7.0),
            positions_xyz_m,
            CENTRE_XY_M,
            ANGLES_RAD,
            LEVEL_FRACTION,
            MAXIMUM_RADIUS_M,
            0.0,
        )
