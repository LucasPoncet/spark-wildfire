import numpy as np
import pytest

from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.terrain.square_grid_mesh import SquareGridMesh

EXTENT_M = 10.0
CELL_SPACING_M = 1.0
MAXIMUM_RATE_OF_SPREAD_M_PER_S = 2.0
SAFETY_FACTOR = 0.9


def build_mesh(use_diagonal_neighbors: bool) -> SquareGridMesh:
    return SquareGridMesh.from_values(
        EXTENT_M, EXTENT_M, CELL_SPACING_M, use_diagonal_neighbors
    )


def test_cfl_with_known_values() -> None:
    time_step_s = compute_maximum_stable_time_step_s(
        build_mesh(False), MAXIMUM_RATE_OF_SPREAD_M_PER_S, SAFETY_FACTOR
    )
    assert time_step_s == pytest.approx(0.45)


def test_raises_on_zero_ros() -> None:
    with pytest.raises(ValueError, match="rate of spread must be positive"):
        compute_maximum_stable_time_step_s(build_mesh(True), 0.0)


def test_diagonal_distance_used_when_smaller() -> None:
    mesh = build_mesh(True)
    valid_distances_m = mesh.neighbor_distances_m[mesh.neighbor_indices >= 0]
    assert np.max(valid_distances_m) == pytest.approx(CELL_SPACING_M * np.sqrt(2.0))
    assert np.min(valid_distances_m) == pytest.approx(CELL_SPACING_M)

    time_step_s = compute_maximum_stable_time_step_s(
        mesh, MAXIMUM_RATE_OF_SPREAD_M_PER_S, SAFETY_FACTOR
    )
    assert time_step_s == pytest.approx(
        SAFETY_FACTOR * CELL_SPACING_M / MAXIMUM_RATE_OF_SPREAD_M_PER_S
    )
    assert time_step_s == compute_maximum_stable_time_step_s(
        build_mesh(False), MAXIMUM_RATE_OF_SPREAD_M_PER_S, SAFETY_FACTOR
    )


def test_time_step_shrinks_as_the_fire_gets_faster() -> None:
    mesh = build_mesh(True)
    assert compute_maximum_stable_time_step_s(
        mesh, 1.0
    ) > compute_maximum_stable_time_step_s(mesh, 2.0)


def test_residence_time_caps_the_timestep() -> None:
    mesh = build_mesh(True)
    slow_fire_time_step_s = compute_maximum_stable_time_step_s(
        mesh, 0.001, SAFETY_FACTOR, residence_time_s=20.0
    )
    assert slow_fire_time_step_s == pytest.approx(SAFETY_FACTOR * 20.0)


def test_residence_time_does_not_loosen_the_crossing_limit() -> None:
    mesh = build_mesh(True)
    crossing_limited_s = compute_maximum_stable_time_step_s(
        mesh, MAXIMUM_RATE_OF_SPREAD_M_PER_S, SAFETY_FACTOR
    )
    assert (
        compute_maximum_stable_time_step_s(
            mesh, MAXIMUM_RATE_OF_SPREAD_M_PER_S, SAFETY_FACTOR, residence_time_s=1e6
        )
        == crossing_limited_s
    )
