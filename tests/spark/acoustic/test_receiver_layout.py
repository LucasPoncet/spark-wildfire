import numpy as np
import pytest

from src.spark.acoustic.receiver_layout import (
    build_grid_receiver_positions_xyz,
    build_random_receiver_positions_xyz,
    build_ring_receiver_positions_xyz,
    compute_collinearity_measure,
    compute_minimum_pairwise_separation_m,
    lift_positions_to_height_xyz,
    validate_receiver_layout,
)

RECEIVER_HEIGHT_M: float = 1.5


def test_a_ring_places_every_receiver_on_the_circle() -> None:
    positions_xyz_m = build_ring_receiver_positions_xyz(
        50.0, 50.0, 45.0, 6, 0.0, RECEIVER_HEIGHT_M
    )
    assert positions_xyz_m.shape == (6, 3)
    radii_m = np.linalg.norm(positions_xyz_m[:, :2] - np.array([50.0, 50.0]), axis=1)
    assert np.allclose(radii_m, 45.0)
    assert np.allclose(positions_xyz_m[:, 2], RECEIVER_HEIGHT_M)


def test_the_ring_start_bearing_rotates_the_whole_array() -> None:
    unrotated = build_ring_receiver_positions_xyz(
        0.0, 0.0, 10.0, 4, 0.0, RECEIVER_HEIGHT_M
    )
    rotated = build_ring_receiver_positions_xyz(
        0.0, 0.0, 10.0, 4, 0.5 * np.pi, RECEIVER_HEIGHT_M
    )
    assert np.allclose(np.roll(unrotated, -1, axis=0), rotated, atol=1e-9)


def test_a_grid_covers_the_extent_at_the_requested_spacing() -> None:
    positions_xyz_m = build_grid_receiver_positions_xyz(
        0.0, 0.0, 100.0, 100.0, 50.0, RECEIVER_HEIGHT_M
    )
    assert positions_xyz_m.shape == (9, 3)
    assert np.allclose(np.unique(positions_xyz_m[:, 0]), [0.0, 50.0, 100.0])


def test_a_random_layout_honours_the_minimum_separation() -> None:
    positions_xyz_m = build_random_receiver_positions_xyz(
        50.0, 50.0, 45.0, 6, 10.0, RECEIVER_HEIGHT_M, 0
    )
    assert positions_xyz_m.shape == (6, 3)
    assert compute_minimum_pairwise_separation_m(positions_xyz_m) >= 10.0


def test_a_random_layout_is_reproducible_from_its_seed() -> None:
    first = build_random_receiver_positions_xyz(
        50.0, 50.0, 45.0, 5, 5.0, RECEIVER_HEIGHT_M, 3
    )
    second = build_random_receiver_positions_xyz(
        50.0, 50.0, 45.0, 5, 5.0, RECEIVER_HEIGHT_M, 3
    )
    assert np.array_equal(first, second)


def test_an_impossible_separation_is_reported() -> None:
    with pytest.raises(ValueError, match="could not place"):
        build_random_receiver_positions_xyz(0.0, 0.0, 1.0, 8, 5.0, RECEIVER_HEIGHT_M, 0)


def test_a_collinear_array_scores_one_and_a_ring_scores_near_zero() -> None:
    collinear_xyz_m = lift_positions_to_height_xyz(
        np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]),
        RECEIVER_HEIGHT_M,
    )
    ring_xyz_m = build_ring_receiver_positions_xyz(
        0.0, 0.0, 20.0, 6, 0.0, RECEIVER_HEIGHT_M
    )
    assert compute_collinearity_measure(collinear_xyz_m) == pytest.approx(1.0, abs=1e-9)
    assert compute_collinearity_measure(ring_xyz_m) < 0.05


def test_two_receivers_are_always_reported_as_collinear() -> None:
    positions_xyz_m = lift_positions_to_height_xyz(
        np.array([[0.0, 0.0], [10.0, 5.0]]), RECEIVER_HEIGHT_M
    )
    assert compute_collinearity_measure(positions_xyz_m) == 1.0


def test_a_collinear_layout_of_three_or_more_receivers_is_rejected() -> None:
    positions_xyz_m = lift_positions_to_height_xyz(
        np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]), RECEIVER_HEIGHT_M
    )
    with pytest.raises(ValueError, match="collinear"):
        validate_receiver_layout(positions_xyz_m, 5.0, 0.98)


def test_a_two_receiver_layout_skips_the_collinearity_guard() -> None:
    positions_xyz_m = lift_positions_to_height_xyz(
        np.array([[30.0, 0.0], [70.0, 0.0]]), RECEIVER_HEIGHT_M
    )
    assert validate_receiver_layout(positions_xyz_m, 5.0, 0.98).shape == (2, 3)


def test_a_crowded_layout_is_rejected() -> None:
    positions_xyz_m = build_ring_receiver_positions_xyz(
        0.0, 0.0, 1.0, 6, 0.0, RECEIVER_HEIGHT_M
    )
    with pytest.raises(ValueError, match="closer than the required"):
        validate_receiver_layout(positions_xyz_m, 5.0, 0.98)


def test_lifting_adds_the_height_column() -> None:
    positions_xyz_m = lift_positions_to_height_xyz(
        np.array([[1.0, 2.0], [3.0, 4.0]]), 0.5
    )
    assert positions_xyz_m.shape == (2, 3)
    assert np.allclose(positions_xyz_m[:, 2], 0.5)


def test_a_single_receiver_has_infinite_separation() -> None:
    assert np.isinf(compute_minimum_pairwise_separation_m(np.array([[0.0, 0.0, 1.5]])))
