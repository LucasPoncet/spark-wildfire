"""Front-shape convergence of the arrival-time engine.

Durations are set by the physics rather than by convenience: pine needle
litter spreads at 2.67 mm/s with no wind and 20.2 mm/s at 2 m/s, so crossing
one 0.5 m cell takes 187 s and 25 s respectively. Runs shorter than that
ignite a handful of cells, and the convex hull of a handful of cells is
degenerate.
"""

import numpy as np
from scipy.spatial import ConvexHull

from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.fire_state import FireState
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_engine import RateOfSpreadEngine
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.terrain.square_grid_mesh import SquareGridMesh

EXTENT_M = 20.0
CELL_SPACING_M = 0.5
WIND_SPEED_M_PER_S = 2.0
FUEL_LOAD_KG_PER_M2 = 1.0

WIND_DRIVEN_DURATION_S = 900.0
CFL_VIOLATION_DURATION_S = 300.0
NO_WIND_DURATION_S = 1800.0


def build_mesh(use_diagonal_neighbors: bool) -> SquareGridMesh:
    return SquareGridMesh.from_values(
        EXTENT_M, EXTENT_M, CELL_SPACING_M, use_diagonal_neighbors
    )


def maximum_rate_of_spread_m_per_s(wind_speed_m_per_s: float) -> float:
    return float(
        compute_rate_of_spread_balbi_2009(
            np.array([wind_speed_m_per_s]),
            np.zeros(1),
            FuelProperties.pine_needle_litter(),
        )[0]
    )


def run_until(
    mesh: SquareGridMesh, wind_speed_m_per_s: float, duration_s: float, dt: float
) -> FireState:
    engine = RateOfSpreadEngine(FuelProperties.pine_needle_litter())
    state = engine.initialize(
        mesh,
        UniformScalarField(FUEL_LOAD_KG_PER_M2),
        ConstantWindField.from_speed_and_bearing(wind_speed_m_per_s, 0.0),
    )
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    state = engine.ignite_cells(state, np.array([center], dtype=np.int64))
    for _ in range(max(1, int(duration_s / dt))):
        state = engine.step(state, dt)
    return state


def ignited_cell_indices(state: FireState) -> set[int]:
    return set(np.flatnonzero(state.has_ignited).tolist())


def symmetric_difference_fraction(first: set[int], second: set[int]) -> float:
    return len(first ^ second) / max(len(first | second), 1)


def convex_hull_area_m2(mesh: SquareGridMesh, state: FireState) -> float:
    ignited_positions_xy_m = mesh.cell_positions_xyz[state.has_ignited, :2]
    return float(ConvexHull(ignited_positions_xy_m).volume)


def test_fire_shape_stable_across_time_steps() -> None:
    mesh = build_mesh(True)
    dt = compute_maximum_stable_time_step_s(
        mesh, maximum_rate_of_spread_m_per_s(WIND_SPEED_M_PER_S)
    )
    shape_fine = ignited_cell_indices(
        run_until(mesh, WIND_SPEED_M_PER_S, WIND_DRIVEN_DURATION_S, dt)
    )
    shape_finer = ignited_cell_indices(
        run_until(mesh, WIND_SPEED_M_PER_S, WIND_DRIVEN_DURATION_S, dt * 0.5)
    )
    assert len(shape_fine) > 100
    assert symmetric_difference_fraction(shape_fine, shape_finer) < 0.05


def test_eight_connectivity_rounder_than_four() -> None:
    dt = compute_maximum_stable_time_step_s(
        build_mesh(True), maximum_rate_of_spread_m_per_s(WIND_SPEED_M_PER_S)
    )
    mesh_four = build_mesh(False)
    mesh_eight = build_mesh(True)
    area_four_m2 = convex_hull_area_m2(
        mesh_four, run_until(mesh_four, 0.0, NO_WIND_DURATION_S, dt)
    )
    area_eight_m2 = convex_hull_area_m2(
        mesh_eight, run_until(mesh_eight, 0.0, NO_WIND_DURATION_S, dt)
    )
    assert area_eight_m2 > area_four_m2 * 1.3


def test_dt_violating_cfl_changes_shape() -> None:
    mesh = build_mesh(True)
    dt = compute_maximum_stable_time_step_s(
        mesh, maximum_rate_of_spread_m_per_s(WIND_SPEED_M_PER_S)
    )
    stable_shape = ignited_cell_indices(
        run_until(mesh, WIND_SPEED_M_PER_S, CFL_VIOLATION_DURATION_S, dt)
    )
    coarse_shape = ignited_cell_indices(
        run_until(mesh, WIND_SPEED_M_PER_S, CFL_VIOLATION_DURATION_S, dt * 3.0)
    )
    assert symmetric_difference_fraction(stable_shape, coarse_shape) > 0.10
