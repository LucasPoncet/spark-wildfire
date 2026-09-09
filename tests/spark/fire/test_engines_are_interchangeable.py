from collections.abc import Callable

import numpy as np
import pytest

from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.cellular_automaton_spread_engine import (
    CellularAutomatonSpreadEngine,
    CellularAutomatonSpreadEngineConfig,
)
from src.spark.fire.fire_state import FireState
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_engine import RateOfSpreadEngine
from src.spark.fire.spread_engine_protocol import SpreadEngineProtocol
from src.spark.terrain.square_grid_mesh import SquareGridMesh

WIND_SPEED_M_PER_S = 3.0
EXTENT_M = 10.0
CELL_SPACING_M = 0.5


def build_cellular_automaton() -> SpreadEngineProtocol:
    return CellularAutomatonSpreadEngine(
        CellularAutomatonSpreadEngineConfig(random_seed=0, burn_duration_s=20.0)
    )


def build_rate_of_spread() -> SpreadEngineProtocol:
    return RateOfSpreadEngine(FuelProperties.pine_needle_litter())


ENGINE_CASES = [
    pytest.param(build_cellular_automaton, 1.0, 8, id="cellular-automaton"),
    pytest.param(build_rate_of_spread, 2.0, 60, id="rate-of-spread"),
]


def run_from_center(
    engine: SpreadEngineProtocol, dt: float, step_count: int
) -> tuple[SquareGridMesh, int, FireState]:
    mesh = SquareGridMesh.from_values(EXTENT_M, EXTENT_M, CELL_SPACING_M, True)
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(WIND_SPEED_M_PER_S, 0.0),
    )
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    state = engine.ignite_cells(state, np.array([center], dtype=np.int64))
    for _ in range(step_count):
        state = engine.step(state, dt=dt)
    return mesh, center, state


@pytest.mark.parametrize(("build_engine", "dt", "step_count"), ENGINE_CASES)
def test_engine_satisfies_the_protocol(
    build_engine: Callable[[], SpreadEngineProtocol], dt: float, step_count: int
) -> None:
    assert isinstance(build_engine(), SpreadEngineProtocol)


@pytest.mark.parametrize(("build_engine", "dt", "step_count"), ENGINE_CASES)
def test_engine_runs_a_fire_through_protocol_calls_only(
    build_engine: Callable[[], SpreadEngineProtocol], dt: float, step_count: int
) -> None:
    engine = build_engine()
    mesh = SquareGridMesh.from_values(EXTENT_M, EXTENT_M, CELL_SPACING_M, True)
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(WIND_SPEED_M_PER_S, 0.0),
    )
    assert not state.has_ignited.any()
    assert state.current_time_s == 0.0

    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    state = engine.ignite_cells(state, np.array([center], dtype=np.int64))
    assert state.is_burning[center]

    for _ in range(step_count):
        state = engine.step(state, dt=dt)

    assert state.current_time_s == pytest.approx(dt * step_count)
    assert state.has_ignited.sum() > 1


@pytest.mark.parametrize(("build_engine", "dt", "step_count"), ENGINE_CASES)
def test_engine_spreads_downwind_further_than_upwind(
    build_engine: Callable[[], SpreadEngineProtocol], dt: float, step_count: int
) -> None:
    mesh, center, state = run_from_center(build_engine(), dt, step_count)
    center_x_m = mesh.cell_positions_xyz[center, 0]
    burnt_x_m = mesh.cell_positions_xyz[state.has_ignited, 0]
    assert burnt_x_m.max() - center_x_m > center_x_m - burnt_x_m.min()


@pytest.mark.parametrize(("build_engine", "dt", "step_count"), ENGINE_CASES)
def test_engine_leaves_part_of_the_grid_unburnt_at_this_timescale(
    build_engine: Callable[[], SpreadEngineProtocol], dt: float, step_count: int
) -> None:
    mesh, _, state = run_from_center(build_engine(), dt, step_count)
    assert 1 < state.has_ignited.sum() < mesh.cell_count


@pytest.mark.parametrize(("build_engine", "dt", "step_count"), ENGINE_CASES)
def test_engine_never_reignites_a_burnt_cell(
    build_engine: Callable[[], SpreadEngineProtocol], dt: float, step_count: int
) -> None:
    engine = build_engine()
    mesh = SquareGridMesh.from_values(EXTENT_M, EXTENT_M, CELL_SPACING_M, True)
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(WIND_SPEED_M_PER_S, 0.0),
    )
    center = (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2
    state = engine.ignite_cells(state, np.array([center], dtype=np.int64))
    previous_has_ignited = state.has_ignited
    for _ in range(step_count):
        state = engine.step(state, dt=dt)
        assert np.all(state.has_ignited >= previous_has_ignited)
        previous_has_ignited = state.has_ignited
