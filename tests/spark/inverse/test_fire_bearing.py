"""The four bearing estimators, on maps whose direction is known by construction.

The skewness test is the important one here, and not because the estimator is
relied on. It is deliberately given a map with the asymmetry the method assumes
— a ring whose downwind half is twice as bright — and required to find it. That
separates two very different statements: the method is wrong, and the method's
precondition is absent. On the fire scene the estimator returns noise, and this
test is what establishes that the reason is the source model radiating every
cell at one amplitude rather than a defect here.
"""

import numpy as np
import pytest

from src.spark.inverse.fire_bearing import (
    BearingEstimate,
    compute_circular_standard_deviation_rad,
    compute_resultant_length,
    estimate_bearing_from_centroid_drift,
    estimate_bearing_from_centroid_offset,
    estimate_bearing_from_principal_axis,
    estimate_bearing_from_skewness,
    estimate_ignition_position_xy_m,
    fuse_bearing_estimates,
)
from src.spark.inverse.map_moments import (
    compute_map_moments,
    compute_skewness_over_directions,
)
from src.spark.inverse.steered_response_power_sequence import (
    SteeredResponsePowerSequence,
)

GRID_SPACING_M: float = 0.5
DOMAIN_EXTENT_M: float = 60.0
CENTRE_XY_M = np.array([30.0, 30.0])
NO_BACKGROUND: float = 0.0
DIRECTION_COUNT: int = 720
BOOTSTRAP_COUNT: int = 60


def build_grid() -> tuple[np.ndarray, tuple[int, int]]:
    axis_m = np.arange(0.5 * GRID_SPACING_M, DOMAIN_EXTENT_M, GRID_SPACING_M)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    positions_xyz_m = np.stack(
        (
            grid_x_m.ravel(),
            grid_y_m.ravel(),
            np.full(grid_x_m.size, 0.5),
        ),
        axis=1,
    )
    return positions_xyz_m, (axis_m.size, axis_m.size)


def build_blob(
    positions_xyz_m: np.ndarray, centre_xy_m: np.ndarray, width_m: float
) -> np.ndarray:
    offsets_m = positions_xyz_m[:, :2] - centre_xy_m
    return np.asarray(
        np.exp(-0.5 * np.sum(offsets_m**2, axis=1) / width_m**2), dtype=np.float64
    )


def build_asymmetric_ring(
    positions_xyz_m: np.ndarray,
    radius_m: float,
    width_m: float,
    bearing_rad: float,
    head_brightness: float,
) -> np.ndarray:
    offsets_m = positions_xyz_m[:, :2] - CENTRE_XY_M
    radii_m = np.linalg.norm(offsets_m, axis=1)
    ring = np.exp(-0.5 * ((radii_m - radius_m) / width_m) ** 2)
    direction = np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
    is_head = offsets_m @ direction > 0.0
    return np.asarray(np.where(is_head, head_brightness * ring, ring), dtype=np.float64)


def build_translating_sequence(bearing_rad: float) -> SteeredResponsePowerSequence:
    positions_xyz_m, grid_shape = build_grid()
    times_s = np.arange(5.0, 55.0, 5.0)
    speed_m_per_s = 0.2
    direction = np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
    maps = np.stack(
        [
            build_blob(
                positions_xyz_m,
                CENTRE_XY_M + speed_m_per_s * (time_s - times_s[0]) * direction,
                3.0,
            )
            for time_s in times_s
        ]
    )
    return SteeredResponsePowerSequence(
        maps=maps,
        times_s=times_s,
        candidate_positions_xyz_m=positions_xyz_m,
        grid_shape=grid_shape,
        grid_spacing_m=GRID_SPACING_M,
    )


@pytest.mark.parametrize("bearing_deg", [0.0, 35.0, 125.0, -160.0])
def test_a_translating_map_gives_its_own_direction(bearing_deg: float) -> None:
    sequence = build_translating_sequence(np.deg2rad(bearing_deg))
    estimate = estimate_bearing_from_centroid_drift(
        sequence, NO_BACKGROUND, None, 1.0, BOOTSTRAP_COUNT, 0
    )
    assert abs(np.rad2deg(estimate.bearing_rad) - bearing_deg) < 1.0


def test_a_map_that_has_not_moved_is_refused_rather_than_guessed() -> None:
    positions_xyz_m, grid_shape = build_grid()
    still = build_blob(positions_xyz_m, CENTRE_XY_M, 3.0)
    sequence = SteeredResponsePowerSequence(
        maps=np.stack([still] * 5),
        times_s=np.arange(5.0, 30.0, 5.0),
        candidate_positions_xyz_m=positions_xyz_m,
        grid_shape=grid_shape,
        grid_spacing_m=GRID_SPACING_M,
    )
    with pytest.raises(ValueError, match="below the"):
        estimate_bearing_from_centroid_drift(
            sequence, NO_BACKGROUND, None, 1.0, BOOTSTRAP_COUNT, 0
        )


def test_too_few_frames_are_refused() -> None:
    positions_xyz_m, grid_shape = build_grid()
    sequence = SteeredResponsePowerSequence(
        maps=np.stack([build_blob(positions_xyz_m, CENTRE_XY_M, 3.0)] * 2),
        times_s=np.array([5.0, 10.0]),
        candidate_positions_xyz_m=positions_xyz_m,
        grid_shape=grid_shape,
        grid_spacing_m=GRID_SPACING_M,
    )
    with pytest.raises(ValueError, match="at least 3 frames"):
        estimate_bearing_from_centroid_drift(
            sequence, NO_BACKGROUND, None, 1.0, BOOTSTRAP_COUNT, 0
        )


def test_the_ignition_point_is_the_first_frame_centroid() -> None:
    sequence = build_translating_sequence(0.0)
    ignition_xy_m = estimate_ignition_position_xy_m(sequence, NO_BACKGROUND, None)
    assert ignition_xy_m == pytest.approx(CENTRE_XY_M, abs=0.2)


@pytest.mark.parametrize("bearing_deg", [0.0, 47.0, 140.0, -95.0])
def test_a_displaced_map_points_away_from_its_origin(bearing_deg: float) -> None:
    positions_xyz_m, _ = build_grid()
    bearing_rad = np.deg2rad(bearing_deg)
    offset_m = 9.0 * np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
    estimate = estimate_bearing_from_centroid_offset(
        build_blob(positions_xyz_m, CENTRE_XY_M + offset_m, 3.0),
        positions_xyz_m,
        CENTRE_XY_M,
        NO_BACKGROUND,
        1.0,
    )
    assert abs(np.rad2deg(estimate.bearing_rad) - bearing_deg) < 1.0


def test_a_map_still_on_its_origin_is_refused() -> None:
    positions_xyz_m, _ = build_grid()
    with pytest.raises(ValueError, match="from the ignition point"):
        estimate_bearing_from_centroid_offset(
            build_blob(positions_xyz_m, CENTRE_XY_M, 3.0),
            positions_xyz_m,
            CENTRE_XY_M,
            NO_BACKGROUND,
            1.0,
        )


@pytest.mark.parametrize("bearing_deg", [0.0, 60.0, 200.0])
def test_skewness_finds_a_brighter_head_when_there_is_one(bearing_deg: float) -> None:
    positions_xyz_m, _ = build_grid()
    bearing_rad = np.deg2rad(bearing_deg)
    estimate = estimate_bearing_from_skewness(
        build_asymmetric_ring(positions_xyz_m, 8.0, 1.5, bearing_rad, 2.0),
        positions_xyz_m,
        NO_BACKGROUND,
        DIRECTION_COUNT,
    )
    error_rad = np.arctan2(
        np.sin(estimate.bearing_rad - bearing_rad),
        np.cos(estimate.bearing_rad - bearing_rad),
    )
    assert abs(np.rad2deg(error_rad)) < 5.0


def test_a_uniform_ring_reports_no_lean_and_an_asymmetric_one_reports_some() -> None:
    """The confidence the estimator attaches to its own answer.

    A ring of one brightness has no lean, so the swept maximum sits at the
    midpoint of the sweep's range and the resultant is a half. A ring whose
    downwind half is brighter has one, and the resultant rises above that. The
    number is what lets a fusion down-weight a frame whose map cannot answer
    the question rather than taking its argmax at face value.
    """
    positions_xyz_m, _ = build_grid()
    uniform = estimate_bearing_from_skewness(
        build_asymmetric_ring(positions_xyz_m, 8.0, 1.5, 0.0, 1.0),
        positions_xyz_m,
        NO_BACKGROUND,
        DIRECTION_COUNT,
    )
    leaning = estimate_bearing_from_skewness(
        build_asymmetric_ring(positions_xyz_m, 8.0, 1.5, 0.0, 2.0),
        positions_xyz_m,
        NO_BACKGROUND,
        DIRECTION_COUNT,
    )
    assert uniform.resultant_length == pytest.approx(0.5, abs=0.02)
    assert leaning.resultant_length > uniform.resultant_length


def test_taking_the_argmax_would_return_the_reverse_bearing() -> None:
    """The sign the plan gets wrong, pinned so it cannot come back.

    A third moment points along a distribution's long tail. Concentrating mass
    toward the head leaves the tail behind it, so the skewness along the true
    bearing is negative and its maximum sits half a turn away.
    """
    positions_xyz_m, _ = build_grid()
    bearing_rad = np.deg2rad(0.0)
    ring = build_asymmetric_ring(positions_xyz_m, 8.0, 1.5, bearing_rad, 2.0)
    moments = compute_map_moments(ring, positions_xyz_m, NO_BACKGROUND)
    angles_rad, skewness = compute_skewness_over_directions(
        ring, positions_xyz_m, moments.centroid_xy_m, DIRECTION_COUNT, NO_BACKGROUND
    )
    argmax_deg = float(np.rad2deg(angles_rad[int(np.argmax(skewness))]))
    argmin_deg = float(np.rad2deg(angles_rad[int(np.argmin(skewness))]))
    assert abs(argmax_deg - 180.0) < 5.0
    assert min(argmin_deg, 360.0 - argmin_deg) < 5.0


def test_the_principal_axis_follows_an_elongated_map() -> None:
    positions_xyz_m, _ = build_grid()
    bearing_rad = np.deg2rad(30.0)
    direction = np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
    across = np.array([-direction[1], direction[0]])
    offsets_m = positions_xyz_m[:, :2] - (CENTRE_XY_M + 6.0 * direction)
    along_m = offsets_m @ direction
    across_m = offsets_m @ across
    elongated = np.exp(-0.5 * ((along_m / 6.0) ** 2 + (across_m / 1.5) ** 2))
    estimate = estimate_bearing_from_principal_axis(
        elongated, positions_xyz_m, CENTRE_XY_M, NO_BACKGROUND, np.zeros((2, 2))
    )
    assert abs(np.rad2deg(estimate.bearing_rad) - 30.0) < 2.0


def test_the_principal_axis_refuses_an_unresolved_front() -> None:
    positions_xyz_m, _ = build_grid()
    blob = build_blob(positions_xyz_m, CENTRE_XY_M + np.array([6.0, 0.0]), 2.0)
    with pytest.raises(ValueError, match="not resolved along any axis"):
        estimate_bearing_from_principal_axis(
            blob,
            positions_xyz_m,
            CENTRE_XY_M,
            NO_BACKGROUND,
            np.diag([1e4, 1e4]),
        )


def test_fusing_agreeing_directions_keeps_them_and_reports_high_resultant() -> None:
    estimates = [
        BearingEstimate(np.deg2rad(30.0), 0.05, "centroid_drift", 10, 0.99),
        BearingEstimate(np.deg2rad(32.0), 0.06, "centroid_offset", 1, 0.98),
    ]
    fused = fuse_bearing_estimates(estimates, {"centroid_drift": 1.0})
    assert 30.0 <= np.rad2deg(fused.bearing_rad) <= 32.0
    assert fused.resultant_length > 0.99


def test_fusing_opposed_directions_reports_a_low_resultant() -> None:
    estimates = [
        BearingEstimate(np.deg2rad(0.0), 0.05, "centroid_drift", 10, 0.99),
        BearingEstimate(np.deg2rad(180.0), 0.05, "skewness", 1, 0.99),
    ]
    fused = fuse_bearing_estimates(estimates, {"centroid_drift": 1.0, "skewness": 1.0})
    assert fused.resultant_length < 0.1


def test_a_zero_weight_keeps_a_method_out_of_the_fusion() -> None:
    estimates = [
        BearingEstimate(np.deg2rad(20.0), 0.05, "centroid_drift", 10, 0.99),
        BearingEstimate(np.deg2rad(170.0), 0.05, "skewness", 1, 0.99),
    ]
    fused = fuse_bearing_estimates(estimates, {"centroid_drift": 1.0, "skewness": 0.0})
    assert abs(np.rad2deg(fused.bearing_rad) - 20.0) < 1.0


def test_fusing_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="no bearing to fuse"):
        fuse_bearing_estimates([], {})


def test_a_tight_cluster_of_angles_has_a_small_circular_spread() -> None:
    tight = compute_resultant_length(np.deg2rad([30.0, 31.0, 29.0]), np.ones(3))
    loose = compute_resultant_length(np.deg2rad([0.0, 120.0, 240.0]), np.ones(3))
    assert compute_circular_standard_deviation_rad(
        tight
    ) < compute_circular_standard_deviation_rad(loose)


def test_a_uniform_direction_reports_the_widest_possible_spread() -> None:
    assert compute_circular_standard_deviation_rad(0.0) == pytest.approx(np.pi)
