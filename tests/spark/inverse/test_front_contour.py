"""Reading a front's outer edge off a density, direction by direction.

The uncovered-direction case is the one that matters. A radiating fire front is
an open arc, so an extractor that returned a distance in every direction would
be inventing three quarters of a curve, and every metric taken against it would
be measuring the invention. What is asserted here is that a direction with no
front reports none.

A clean arc is not the hard case, and the halo fixture below is why. A level is
a fraction of the density's global peak, so any dim residue about the origin —
which is what a deconvolution actually leaves — crosses that level in *every*
direction and hands back a radius belonging to the arc's brightness rather than
to the direction it was read in. The pair of tests around
`build_arc_with_halo` states both halves: ungated, the extractor claims the
whole circle; gated on wedge mass, it claims the arc.
"""

import numpy as np
import pytest

from src.spark.inverse.front_contour import (
    compute_angular_coverage_fraction,
    compute_angular_support,
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


def build_arc_with_halo(
    positions_xyz_m: np.ndarray,
    radius_m: float,
    half_width_rad: float,
    halo_amplitude: float,
) -> np.ndarray:
    """An arc, plus the dim residue a deconvolution leaves about the origin."""
    radii_m = np.linalg.norm(positions_xyz_m[:, :2] - CENTRE_XY_M, axis=1)
    halo = halo_amplitude * np.exp(-0.5 * (radii_m / 1.5) ** 2)
    return np.asarray(
        build_arc(positions_xyz_m, radius_m, half_width_rad) + halo, dtype=np.float64
    )


def extract(
    density: np.ndarray,
    support_fraction: float = 0.0,
    local_level_fraction: float = 0.0,
) -> np.ndarray:
    return compute_contour_distance_by_angle(
        density,
        build_grid(),
        CENTRE_XY_M,
        ANGLES_RAD,
        LEVEL_FRACTION,
        local_level_fraction,
        support_fraction,
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
        0.0,
        0.0,
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
        0.0,
        0.0,
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
        0.0,
        0.0,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    high = compute_contour_distance_by_angle(
        density,
        positions_xyz_m,
        CENTRE_XY_M,
        ANGLES_RAD,
        0.8,
        0.0,
        0.0,
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
            0.0,
            0.0,
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
            0.0,
            0.0,
            MAXIMUM_RADIUS_M,
            0.0,
        )


def test_the_best_direction_carries_full_support() -> None:
    support = compute_angular_support(
        build_arc(build_grid(), 8.0, np.deg2rad(45.0)),
        build_grid(),
        CENTRE_XY_M,
        ANGLES_RAD,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    assert float(np.max(support)) == pytest.approx(1.0)
    assert float(np.min(support)) == pytest.approx(0.0, abs=1e-9)
    assert float(support[0]) > 0.9


def test_support_weighs_far_ground_above_near_ground() -> None:
    """Why the wedge mass carries a radius weight rather than being a line sum.

    Two rings of the same brightness and width, one at four metres ahead and
    one at eight metres behind. A plain line integral would call them equal;
    the wedge they sweep does not, and the further one wins.
    """
    positions_xyz_m = build_grid()
    near = build_arc(positions_xyz_m, 4.0, np.deg2rad(30.0))
    offsets_m = positions_xyz_m[:, :2] - CENTRE_XY_M
    far_ring = np.exp(-0.5 * ((np.linalg.norm(offsets_m, axis=1) - 8.0) / 0.8) ** 2)
    behind = np.abs(np.arctan2(offsets_m[:, 1], offsets_m[:, 0])) >= np.deg2rad(150.0)
    support = compute_angular_support(
        near + np.where(behind, far_ring, 0.0),
        positions_xyz_m,
        CENTRE_XY_M,
        ANGLES_RAD,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    ahead_index = 0
    behind_index = int(ANGLES_RAD.size // 2)
    assert float(support[behind_index]) > float(support[ahead_index])


def test_an_arc_with_a_halo_claims_every_direction_ungated() -> None:
    """The failure the gate exists for, stated before it is fixed."""
    distances_m = extract(build_arc_with_halo(build_grid(), 8.0, np.deg2rad(45.0), 0.6))
    assert compute_angular_coverage_fraction(distances_m) == pytest.approx(1.0)


def test_the_support_gate_gives_that_arc_its_sector_back() -> None:
    half_width_rad = np.deg2rad(45.0)
    distances_m = extract(
        build_arc_with_halo(build_grid(), 8.0, half_width_rad, 0.6), 0.25
    )
    covered = compute_angular_coverage_fraction(distances_m)
    assert covered == pytest.approx(2.0 * half_width_rad / (2.0 * np.pi), abs=0.06)


def test_the_gate_keeps_the_radii_it_does_not_refuse() -> None:
    """The gate refuses directions; it is not allowed to move a radius."""
    density = build_arc(build_grid(), 8.0, np.deg2rad(45.0))
    ungated = extract(density)
    gated = extract(density, 0.25)
    kept = np.isfinite(gated)
    assert bool(np.any(kept))
    assert gated[kept] == pytest.approx(ungated[kept])


def test_the_gate_leaves_a_closed_shape_closed() -> None:
    """A front that really does surround the origin must survive the gate."""
    distances_m = extract(build_disc(build_grid(), 7.0), 0.25)
    assert compute_angular_coverage_fraction(distances_m) == pytest.approx(1.0)


def test_a_support_fraction_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="support fraction"):
        extract(build_disc(build_grid(), 7.0), 1.4)


def build_ragged_ring(
    positions_xyz_m: np.ndarray, radius_m: float, dim_amplitude: float
) -> np.ndarray:
    """A closed ring, bright on one side and dim on the other.

    What a probabilistic spread model actually produces: the front surrounds
    the origin but does not radiate evenly around it.
    """
    offsets_m = positions_xyz_m[:, :2] - CENTRE_XY_M
    radii_m = np.linalg.norm(offsets_m, axis=1)
    angles_rad = np.arctan2(offsets_m[:, 1], offsets_m[:, 0])
    ring = np.exp(-0.5 * ((radii_m - radius_m) / 0.8) ** 2)
    bright = np.abs(np.arctan2(np.sin(angles_rad), np.cos(angles_rad))) <= 0.5 * np.pi
    return np.asarray(ring * np.where(bright, 1.0, dim_amplitude), dtype=np.float64)


def extract_with_levels(
    density: np.ndarray, level_fraction: float, local_level_fraction: float
) -> np.ndarray:
    return compute_contour_distance_by_angle(
        density,
        build_grid(),
        CENTRE_XY_M,
        ANGLES_RAD,
        level_fraction,
        local_level_fraction,
        0.0,
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )


def test_a_global_level_loses_the_dim_half_of_a_closed_front() -> None:
    """The failure measured on configs/f2, stated before it is fixed."""
    density = build_ragged_ring(build_grid(), 8.0, 0.08)
    covered = compute_angular_coverage_fraction(extract_with_levels(density, 0.5, 0.0))
    assert covered == pytest.approx(0.5, abs=0.08)


def test_a_per_ray_level_gives_the_dim_half_back() -> None:
    density = build_ragged_ring(build_grid(), 8.0, 0.08)
    covered = compute_angular_coverage_fraction(extract_with_levels(density, 0.02, 0.5))
    assert covered == pytest.approx(1.0, abs=0.03)


def test_a_per_ray_level_finds_the_dim_half_at_the_same_radius() -> None:
    """Dimmer, not nearer — the radius must not depend on the brightness."""
    distances_m = extract_with_levels(
        build_ragged_ring(build_grid(), 8.0, 0.08), 0.02, 0.5
    )
    wrapped_rad = np.arctan2(np.sin(ANGLES_RAD), np.cos(ANGLES_RAD))
    bright = np.abs(wrapped_rad) <= 0.4 * np.pi
    dim = np.abs(wrapped_rad) >= 0.6 * np.pi
    assert float(np.nanmedian(distances_m[dim])) == pytest.approx(
        float(np.nanmedian(distances_m[bright])), abs=0.3
    )


def test_the_global_level_still_floors_a_direction_carrying_only_a_halo() -> None:
    """The per-ray level cannot be allowed to promote noise into a front."""
    positions_xyz_m = build_grid()
    density = build_arc_with_halo(positions_xyz_m, 8.0, np.deg2rad(45.0), 0.01)
    covered = compute_angular_coverage_fraction(extract_with_levels(density, 0.05, 0.5))
    assert covered == pytest.approx(0.25, abs=0.08)


def test_a_local_level_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="local contour level fraction"):
        extract_with_levels(build_disc(build_grid(), 7.0), 0.5, 1.4)
