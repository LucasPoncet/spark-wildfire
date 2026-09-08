import numpy as np
import pytest

from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.fire_state import FireState
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_engine import RateOfSpreadEngine
from src.spark.fire.rate_of_spread_equations import compute_base_rate_of_spread_m_per_s
from src.spark.fire.spread_engine_protocol import SpreadEngineProtocol
from src.spark.terrain.square_grid_mesh import SquareGridMesh

CELL_SPACING_M = 0.5
WIND_SPEED_M_PER_S = 3.0
FUEL_LOAD_KG_PER_M2 = 1.0


def build_run(
    cells_per_side: int,
    wind_speed_m_per_s: float = WIND_SPEED_M_PER_S,
    wind_bearing_rad: float = 0.0,
    use_diagonal_neighbors: bool = False,
) -> tuple[SquareGridMesh, RateOfSpreadEngine, FireState]:
    extent_m = (cells_per_side - 1) * CELL_SPACING_M
    mesh = SquareGridMesh.from_values(
        extent_m, extent_m, CELL_SPACING_M, use_diagonal_neighbors
    )
    engine = RateOfSpreadEngine(FuelProperties.pine_needle_litter())
    state = engine.initialize(
        mesh,
        UniformScalarField(FUEL_LOAD_KG_PER_M2),
        ConstantWindField.from_speed_and_bearing(wind_speed_m_per_s, wind_bearing_rad),
    )
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    state = engine.ignite_cells(state, np.array([center], dtype=np.int64))
    return mesh, engine, state


def test_engine_satisfies_spread_engine_protocol() -> None:
    assert isinstance(
        RateOfSpreadEngine(FuelProperties.pine_needle_litter()), SpreadEngineProtocol
    )


def test_step_before_initialize_raises_a_named_error() -> None:
    engine = RateOfSpreadEngine(FuelProperties.pine_needle_litter())
    state = FireState(
        ignition_times_s=np.full(4, np.inf),
        burnout_times_s=np.full(4, np.inf),
        is_burning=np.zeros(4, dtype=np.bool_),
        has_ignited=np.zeros(4, dtype=np.bool_),
        current_time_s=0.0,
    )
    with pytest.raises(RuntimeError, match="initialize must be called"):
        engine.step(state, dt=1.0)


def test_downwind_neighbor_ignites_before_the_upwind_one() -> None:
    mesh, engine, state = build_run(5)
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    for _ in range(400):
        state = engine.step(state, dt=1.0)
    downwind = center + 1
    upwind = center - 1
    assert state.has_ignited[downwind]
    assert state.has_ignited[upwind]
    assert state.ignition_times_s[downwind] < state.ignition_times_s[upwind]


def test_crosswind_neighbors_ignite_at_the_same_time() -> None:
    mesh, engine, state = build_run(5)
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    for _ in range(400):
        state = engine.step(state, dt=1.0)
    above = center + mesh.n_x
    below = center - mesh.n_x
    assert state.ignition_times_s[above] == pytest.approx(state.ignition_times_s[below])


def test_upwind_neighbor_ignites_at_the_base_rate_of_spread() -> None:
    mesh, engine, state = build_run(5)
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    for _ in range(400):
        state = engine.step(state, dt=1.0)
    base_rate = compute_base_rate_of_spread_m_per_s(FuelProperties.pine_needle_litter())
    assert state.ignition_times_s[center - 1] == pytest.approx(
        CELL_SPACING_M / base_rate, rel=1e-9
    )


def test_front_is_elliptical_after_fifty_steps() -> None:
    mesh, engine, state = build_run(11, use_diagonal_neighbors=True)
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    for _ in range(50):
        state = engine.step(state, dt=2.0)
    center_xyz = mesh.cell_positions_xyz[center]
    burnt_xyz = mesh.cell_positions_xyz[state.has_ignited]
    downwind_reach_m = burnt_xyz[:, 0].max() - center_xyz[0]
    upwind_reach_m = center_xyz[0] - burnt_xyz[:, 0].min()
    crosswind_reach_m = burnt_xyz[:, 1].max() - center_xyz[1]
    assert downwind_reach_m > crosswind_reach_m > upwind_reach_m
    assert downwind_reach_m > 2.0 * upwind_reach_m


def test_front_is_symmetric_without_wind() -> None:
    mesh, engine, state = build_run(11, wind_speed_m_per_s=0.0)
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    for _ in range(60):
        state = engine.step(state, dt=5.0)
    center_xyz = mesh.cell_positions_xyz[center]
    burnt_xyz = mesh.cell_positions_xyz[state.has_ignited]
    assert burnt_xyz[:, 0].max() - center_xyz[0] == pytest.approx(
        center_xyz[0] - burnt_xyz[:, 0].min()
    )
    assert burnt_xyz[:, 1].max() - center_xyz[1] == pytest.approx(
        center_xyz[0] - burnt_xyz[:, 0].min()
    )


def test_ignition_times_barely_depend_on_the_timestep() -> None:
    def run(dt: float, step_count: int) -> np.ndarray:
        _, engine, state = build_run(9)
        for _ in range(step_count):
            state = engine.step(state, dt=dt)
        return state.ignition_times_s

    coarse = run(10.0, 40)
    fine = run(1.0, 400)
    both_ignited = np.isfinite(coarse) & np.isfinite(fine)
    np.testing.assert_allclose(coarse[both_ignited], fine[both_ignited], rtol=1e-9)


def test_cells_burn_out_after_the_fuel_residence_time() -> None:
    fuel = FuelProperties.pine_needle_litter()
    mesh, engine, state = build_run(5)
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    assert state.is_burning[center]
    assert state.burnout_times_s[center] == pytest.approx(fuel.residence_time_s)
    for _ in range(int(fuel.residence_time_s) + 1):
        state = engine.step(state, dt=1.0)
    assert not state.is_burning[center]
    assert state.has_ignited[center]


def test_fire_does_not_spread_without_fuel() -> None:
    mesh = SquareGridMesh.from_values(2.0, 2.0, CELL_SPACING_M, False)
    engine = RateOfSpreadEngine(FuelProperties.pine_needle_litter())
    state = engine.initialize(
        mesh,
        UniformScalarField(0.0),
        ConstantWindField.from_speed_and_bearing(WIND_SPEED_M_PER_S, 0.0),
    )
    state = engine.ignite_cells(state, np.array([0], dtype=np.int64))
    for _ in range(100):
        state = engine.step(state, dt=10.0)
    assert not state.has_ignited.any()
