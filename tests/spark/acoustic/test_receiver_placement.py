import numpy as np
import pytest

from src.config.receiver_configuration import ReceiverConfiguration
from src.spark.acoustic.receiver_placement import (
    place_receivers_from_configuration,
    place_receivers_in_grid,
    place_receivers_in_ring,
    place_receivers_randomly,
)

CENTER_X_M = 50.0
CENTER_Y_M = 50.0
RADIUS_M = 40.0
GRID_EXTENT_M = 100.0


def distances_from_center_m(positions_xy_m: np.ndarray) -> np.ndarray:
    return np.linalg.norm(positions_xy_m - np.array([CENTER_X_M, CENTER_Y_M]), axis=1)


def test_ring_placement_has_correct_count_and_radius() -> None:
    positions_xy_m = place_receivers_in_ring(CENTER_X_M, CENTER_Y_M, RADIUS_M, 8)
    assert positions_xy_m.shape == (8, 2)
    np.testing.assert_allclose(
        distances_from_center_m(positions_xy_m), RADIUS_M, atol=1e-10
    )


def test_ring_placement_starts_on_the_positive_x_axis() -> None:
    positions_xy_m = place_receivers_in_ring(CENTER_X_M, CENTER_Y_M, RADIUS_M, 4)
    np.testing.assert_allclose(
        positions_xy_m[0], [CENTER_X_M + RADIUS_M, CENTER_Y_M], atol=1e-10
    )


def test_ring_placement_rejects_zero_receivers() -> None:
    with pytest.raises(ValueError, match="receiver count must be positive"):
        place_receivers_in_ring(CENTER_X_M, CENTER_Y_M, RADIUS_M, 0)


def test_grid_placement_covers_extent() -> None:
    positions_xy_m = place_receivers_in_grid(
        0.0, 0.0, GRID_EXTENT_M, GRID_EXTENT_M, 10.0
    )
    assert positions_xy_m.shape == (121, 2)
    assert positions_xy_m[:, 0].min() == 0.0
    assert positions_xy_m[:, 0].max() == GRID_EXTENT_M
    assert positions_xy_m[:, 1].min() == 0.0
    assert positions_xy_m[:, 1].max() == GRID_EXTENT_M


def test_grid_placement_rejects_non_positive_spacing() -> None:
    with pytest.raises(ValueError, match="spacing must be positive"):
        place_receivers_in_grid(0.0, 0.0, GRID_EXTENT_M, GRID_EXTENT_M, 0.0)


def test_random_placement_stays_inside_circle() -> None:
    positions_xy_m = place_receivers_randomly(CENTER_X_M, CENTER_Y_M, RADIUS_M, 100)
    assert positions_xy_m.shape == (100, 2)
    assert np.all(distances_from_center_m(positions_xy_m) <= RADIUS_M)


def test_random_placement_is_reproducible_per_seed() -> None:
    first = place_receivers_randomly(CENTER_X_M, CENTER_Y_M, RADIUS_M, 20, seed=3)
    second = place_receivers_randomly(CENTER_X_M, CENTER_Y_M, RADIUS_M, 20, seed=3)
    different = place_receivers_randomly(CENTER_X_M, CENTER_Y_M, RADIUS_M, 20, seed=4)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_dispatcher_routes_to_ring() -> None:
    positions_xy_m = place_receivers_from_configuration(
        ReceiverConfiguration(
            placement_strategy="ring", receiver_count=6, ring_radius_m=30.0
        ),
        GRID_EXTENT_M,
        GRID_EXTENT_M,
    )
    assert positions_xy_m.shape == (6, 2)
    np.testing.assert_allclose(
        distances_from_center_m(positions_xy_m), 30.0, atol=1e-10
    )


def test_dispatcher_routes_to_grid() -> None:
    positions_xy_m = place_receivers_from_configuration(
        ReceiverConfiguration(placement_strategy="grid", grid_spacing_m=10.0),
        GRID_EXTENT_M,
        GRID_EXTENT_M,
    )
    assert positions_xy_m.shape == (121, 2)


def test_dispatcher_routes_to_random() -> None:
    positions_xy_m = place_receivers_from_configuration(
        ReceiverConfiguration(
            placement_strategy="random", receiver_count=12, random_radius_m=RADIUS_M
        ),
        GRID_EXTENT_M,
        GRID_EXTENT_M,
    )
    assert positions_xy_m.shape == (12, 2)
    assert np.all(distances_from_center_m(positions_xy_m) <= RADIUS_M)


def test_dispatcher_resolves_fractional_centre_against_the_grid() -> None:
    positions_xy_m = place_receivers_from_configuration(
        ReceiverConfiguration(
            placement_strategy="ring",
            receiver_count=4,
            ring_radius_m=10.0,
            ring_center_x_fraction=0.25,
            ring_center_y_fraction=0.75,
        ),
        200.0,
        400.0,
    )
    np.testing.assert_allclose(positions_xy_m.mean(axis=0), [50.0, 300.0], atol=1e-10)


def test_dispatcher_raises_on_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown receiver placement strategy"):
        place_receivers_from_configuration(
            ReceiverConfiguration(placement_strategy="hexagonal"),
            GRID_EXTENT_M,
            GRID_EXTENT_M,
        )
