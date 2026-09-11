"""The ground truth every estimate is scored against, on hand-built masks.

If the truth extractor puts the head at the wrong end, every bearing error in
the report is wrong by half a turn and every rate carries the wrong sign, with
nothing anywhere to contradict it. So the head, the back and the bearing are
each checked on a mask whose answer can be read off by eye.
"""

import numpy as np
import pytest

from src.utils.metrics.fire_front_ground_truth import (
    compute_hull_area_m2,
    compute_principal_semi_axes_m,
    compute_true_bearing_from_sequence_rad,
    extract_fire_front_truth,
)

IGNITION_XY_M = np.array([30.0, 30.0])
CELL_AREA_M2: float = 0.25
SHAPE_FACTOR: float = float(np.sqrt(2.0))


def build_arc(bearing_rad: float, radius_m: float, point_count: int) -> np.ndarray:
    """Half a ring, centred on the spread direction, as a burnt-out front is."""
    offsets_rad = np.linspace(-0.5 * np.pi, 0.5 * np.pi, point_count)
    angles_rad = bearing_rad + offsets_rad
    return np.stack(
        (
            IGNITION_XY_M[0] + radius_m * np.cos(angles_rad),
            IGNITION_XY_M[1] + radius_m * np.sin(angles_rad),
        ),
        axis=1,
    )


@pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 130.0, -70.0])
def test_the_bearing_points_from_ignition_to_the_head(bearing_deg: float) -> None:
    bearing_rad = np.deg2rad(bearing_deg)
    truth = extract_fire_front_truth(
        50.0,
        build_arc(bearing_rad, 10.0, 51),
        IGNITION_XY_M,
        CELL_AREA_M2,
        SHAPE_FACTOR,
    )
    error_rad = np.arctan2(
        np.sin(truth.bearing_rad - bearing_rad),
        np.cos(truth.bearing_rad - bearing_rad),
    )
    assert abs(np.rad2deg(error_rad)) < 2.0


def test_the_head_is_the_furthest_cell_along_the_spread_direction() -> None:
    positions_xy_m = build_arc(0.0, 10.0, 51)
    truth = extract_fire_front_truth(
        50.0, positions_xy_m, IGNITION_XY_M, CELL_AREA_M2, SHAPE_FACTOR
    )
    assert truth.head_position_xy_m[0] == pytest.approx(
        float(np.max(positions_xy_m[:, 0])), abs=1e-9
    )


def test_the_back_is_the_furthest_cell_against_it() -> None:
    positions_xy_m = build_arc(0.0, 10.0, 51)
    truth = extract_fire_front_truth(
        50.0, positions_xy_m, IGNITION_XY_M, CELL_AREA_M2, SHAPE_FACTOR
    )
    assert truth.back_position_xy_m[0] == pytest.approx(
        float(np.min(positions_xy_m[:, 0])), abs=1e-9
    )


def test_the_head_lies_further_out_than_the_back() -> None:
    truth = extract_fire_front_truth(
        50.0, build_arc(0.0, 10.0, 51), IGNITION_XY_M, CELL_AREA_M2, SHAPE_FACTOR
    )
    assert truth.head_distance_m > truth.back_distance_m


def test_a_hand_built_three_cell_mask_gives_its_own_head_and_back() -> None:
    positions_xy_m = np.array([[31.0, 30.0], [33.0, 30.0], [35.0, 30.0]])
    truth = extract_fire_front_truth(
        10.0, positions_xy_m, IGNITION_XY_M, CELL_AREA_M2, SHAPE_FACTOR
    )
    assert truth.head_position_xy_m == pytest.approx([35.0, 30.0])
    assert truth.back_position_xy_m == pytest.approx([31.0, 30.0])
    assert truth.head_distance_m == pytest.approx(5.0)
    assert truth.back_distance_m == pytest.approx(1.0)
    assert np.rad2deg(truth.bearing_rad) == pytest.approx(0.0, abs=1e-9)


def test_the_centroid_is_the_mean_of_the_active_cells() -> None:
    positions_xy_m = np.array([[31.0, 30.0], [33.0, 30.0], [35.0, 34.0]])
    truth = extract_fire_front_truth(
        10.0, positions_xy_m, IGNITION_XY_M, CELL_AREA_M2, SHAPE_FACTOR
    )
    assert truth.front_centroid_xy_m == pytest.approx(np.mean(positions_xy_m, axis=0))


def test_an_observation_with_no_active_cell_is_rejected() -> None:
    with pytest.raises(ValueError, match="no active cell"):
        extract_fire_front_truth(
            5.0, np.empty((0, 2)), IGNITION_XY_M, CELL_AREA_M2, SHAPE_FACTOR
        )


def test_a_ring_recovers_its_radius_through_the_shape_factor() -> None:
    radius_m = 7.0
    angles_rad = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    ring_xy_m = np.stack(
        (
            IGNITION_XY_M[0] + radius_m * np.cos(angles_rad),
            IGNITION_XY_M[1] + radius_m * np.sin(angles_rad),
        ),
        axis=1,
    )
    semi_axes_m = compute_principal_semi_axes_m(ring_xy_m, SHAPE_FACTOR)
    assert semi_axes_m[0] == pytest.approx(radius_m, rel=1e-6)


def test_a_single_cell_has_no_extent() -> None:
    assert compute_principal_semi_axes_m(
        np.array([[30.0, 30.0]]), SHAPE_FACTOR
    ) == pytest.approx([0.0, 0.0])


def test_a_square_encloses_its_own_area() -> None:
    square_xy_m = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    assert compute_hull_area_m2(square_xy_m, CELL_AREA_M2) == pytest.approx(16.0)


def test_collinear_cells_fall_back_to_counting_rather_than_raising() -> None:
    collinear_xy_m = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert compute_hull_area_m2(collinear_xy_m, CELL_AREA_M2) == pytest.approx(
        4.0 * CELL_AREA_M2
    )


def test_two_cells_fall_back_to_counting() -> None:
    assert compute_hull_area_m2(
        np.array([[0.0, 0.0], [1.0, 0.0]]), CELL_AREA_M2
    ) == pytest.approx(2.0 * CELL_AREA_M2)


def test_the_equivalent_radius_matches_the_hull_it_came_from() -> None:
    radius_m = 6.0
    angles_rad = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    disc_xy_m = np.stack(
        (
            IGNITION_XY_M[0] + radius_m * np.cos(angles_rad),
            IGNITION_XY_M[1] + radius_m * np.sin(angles_rad),
        ),
        axis=1,
    )
    truth = extract_fire_front_truth(
        50.0, disc_xy_m, IGNITION_XY_M, CELL_AREA_M2, SHAPE_FACTOR
    )
    assert truth.equivalent_radius_m == pytest.approx(radius_m, rel=0.01)


def test_a_sequence_bearing_comes_from_its_last_observation() -> None:
    truths = [
        extract_fire_front_truth(
            float(time_s),
            build_arc(np.deg2rad(40.0), 2.0 + 0.2 * float(time_s), 31),
            IGNITION_XY_M,
            CELL_AREA_M2,
            SHAPE_FACTOR,
        )
        for time_s in (10.0, 30.0, 50.0)
    ]
    assert np.rad2deg(compute_true_bearing_from_sequence_rad(truths)) == pytest.approx(
        40.0, abs=2.0
    )


def test_an_empty_sequence_has_no_bearing() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        compute_true_bearing_from_sequence_rad([])
