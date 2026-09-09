import numpy as np
import pytest

from src.spark.acoustic.burning_cell_source_model import (
    compute_fire_front_mask,
    extract_burning_cell_sources,
    extract_fire_front_sources,
)
from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.fire_state import FireState
from src.spark.fire.spread_engine_protocol import SpreadEngineProtocol
from src.spark.fire.static_source_spread_engine import StaticSourceSpreadEngine
from src.spark.terrain.square_grid_mesh import SquareGridMesh

EXTENT_M = 10.0
CELL_SPACING_M = 1.0
FUEL_LOAD_KG_PER_M2 = 0.5
RESIDENCE_TIME_S = 20.0
TIME_STEP_S = 5.0


def build_mesh() -> SquareGridMesh:
    return SquareGridMesh.from_values(EXTENT_M, EXTENT_M, CELL_SPACING_M, True)


def build_engine_and_state(
    source_cell_indices: tuple[int, ...],
) -> tuple[StaticSourceSpreadEngine, SquareGridMesh, FireState]:
    mesh = build_mesh()
    engine = StaticSourceSpreadEngine()
    state = engine.initialize(
        mesh,
        UniformScalarField(FUEL_LOAD_KG_PER_M2),
        ConstantWindField.from_speed_and_bearing(2.0, 0.0),
    )
    state = engine.ignite_cells(state, np.array(source_cell_indices, dtype=np.int64))
    return engine, mesh, state


def test_the_engine_satisfies_the_spread_engine_protocol() -> None:
    assert isinstance(StaticSourceSpreadEngine(), SpreadEngineProtocol)


def test_initialize_returns_an_unlit_state() -> None:
    mesh = build_mesh()
    state = StaticSourceSpreadEngine().initialize(
        mesh,
        UniformScalarField(FUEL_LOAD_KG_PER_M2),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    assert not state.is_burning.any()
    assert not state.has_ignited.any()
    assert state.current_time_s == 0.0


def test_step_before_initialize_raises_a_named_error() -> None:
    engine = StaticSourceSpreadEngine()
    state = FireState(
        ignition_times_s=np.full(4, np.inf),
        burnout_times_s=np.full(4, np.inf),
        is_burning=np.zeros(4, dtype=np.bool_),
        has_ignited=np.zeros(4, dtype=np.bool_),
        current_time_s=0.0,
    )
    with pytest.raises(RuntimeError, match="initialize must be called"):
        engine.step(state, TIME_STEP_S)


def test_sources_stay_burning_across_steps() -> None:
    engine, _, state = build_engine_and_state((12, 47))
    for _ in range(50):
        state = engine.step(state, TIME_STEP_S)
    assert np.flatnonzero(state.is_burning).tolist() == [12, 47]
    assert np.flatnonzero(state.has_ignited).tolist() == [12, 47]


def test_the_clock_advances_by_the_timestep() -> None:
    engine, _, state = build_engine_and_state((12,))
    for expected_step in range(1, 6):
        state = engine.step(state, TIME_STEP_S)
        assert state.current_time_s == pytest.approx(expected_step * TIME_STEP_S)


def test_the_fire_never_spreads_to_a_neighbour() -> None:
    engine, _, state = build_engine_and_state((60,))
    for _ in range(200):
        state = engine.step(state, TIME_STEP_S)
    assert int(state.is_burning.sum()) == 1


def test_burnout_is_infinite() -> None:
    _, _, state = build_engine_and_state((12,))
    assert bool(np.isinf(state.burnout_times_s).all())


def test_the_returned_state_is_immutable() -> None:
    engine, _, state = build_engine_and_state((12,))
    state = engine.step(state, TIME_STEP_S)
    with pytest.raises(ValueError, match="read-only"):
        state.is_burning[0] = True


def test_a_permanently_burning_cell_stays_on_the_front() -> None:
    engine, mesh, state = build_engine_and_state((60,))
    for _ in range(20):
        state = engine.step(state, TIME_STEP_S)
    is_on_front = compute_fire_front_mask(state, mesh.neighbor_indices)
    assert np.flatnonzero(is_on_front).tolist() == [60]


def test_both_emission_models_agree_on_a_static_source() -> None:
    engine, mesh, state = build_engine_and_state((25, 70))
    state = engine.step(state, TIME_STEP_S)
    front = extract_fire_front_sources(
        state,
        mesh.cell_positions_xyz,
        mesh.neighbor_indices,
        FUEL_LOAD_KG_PER_M2,
        RESIDENCE_TIME_S,
    )
    burning = extract_burning_cell_sources(
        state, mesh.cell_positions_xyz, FUEL_LOAD_KG_PER_M2, RESIDENCE_TIME_S
    )
    np.testing.assert_array_equal(
        front.burning_cell_indices, burning.burning_cell_indices
    )


def test_igniting_a_cell_twice_does_not_move_its_ignition_time() -> None:
    engine, _, state = build_engine_and_state((12,))
    first_ignition_time_s = state.ignition_times_s[12]
    state = engine.step(state, TIME_STEP_S)
    state = engine.ignite_cells(state, np.array([12], dtype=np.int64))
    assert state.ignition_times_s[12] == first_ignition_time_s
