import numpy as np
import pytest

from spark.fields.constant_wind_field import ConstantWindField
from spark.fields.uniform_scalar_field import UniformScalarField
from spark.fire.cellular_automaton_spread_engine import (
    CellularAutomatonSpreadEngine,
    CellularAutomatonSpreadEngineConfig,
)
from spark.fire.fire_state import FireState
from spark.fire.spread_engine_protocol import SpreadEngineProtocol
from spark.terrain.square_grid_mesh import SquareGridMesh

GRID_EXTENT_M = 40.0
CELL_SPACING_M = 1.0
WIND_SPEED_M_PER_S = 5.0
BURN_DURATION_S = 20.0


def build_mesh() -> SquareGridMesh:
    return SquareGridMesh.from_values(
        extent_x_m=GRID_EXTENT_M,
        extent_y_m=GRID_EXTENT_M,
        cell_spacing_m=CELL_SPACING_M,
        use_diagonal_neighbors=True,
    )


def build_engine(seed: int) -> CellularAutomatonSpreadEngine:
    return CellularAutomatonSpreadEngine(
        CellularAutomatonSpreadEngineConfig(
            random_seed=seed, burn_duration_s=BURN_DURATION_S
        )
    )


def center_cell_index(mesh: SquareGridMesh) -> int:
    return (mesh.n_y // 2) * mesh.n_x + mesh.n_x // 2


def burnt_centroid_offset_x_m(wind_bearing_rad: float, seed: int) -> float:
    mesh = build_mesh()
    engine = build_engine(seed)
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(WIND_SPEED_M_PER_S, wind_bearing_rad),
    )
    center = center_cell_index(mesh)
    state = engine.ignite_cells(state, np.array([center], dtype=np.int64))
    for _ in range(20):
        state = engine.step(state, dt=1.0)
    burnt_x_m = mesh.cell_positions_xyz[state.has_ignited, 0]
    return float(burnt_x_m.mean() - mesh.cell_positions_xyz[center, 0])


def test_fire_spreads_downwind_not_upwind() -> None:
    offsets = [burnt_centroid_offset_x_m(0.0, seed) for seed in range(4)]
    assert np.mean(offsets) > 1.0


def test_wind_reversal_reverses_the_front() -> None:
    towards_plus_x = np.mean([burnt_centroid_offset_x_m(0.0, s) for s in range(4)])
    towards_minus_x = np.mean([burnt_centroid_offset_x_m(np.pi, s) for s in range(4)])
    assert towards_plus_x > 0.0
    assert towards_minus_x < 0.0


def test_zero_wind_front_is_symmetric() -> None:
    offsets = [burnt_centroid_offset_x_m(0.0, seed) for seed in range(4)]
    mesh = build_mesh()
    engine = build_engine(0)
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    center = center_cell_index(mesh)
    state = engine.ignite_cells(state, np.array([center], dtype=np.int64))
    for _ in range(20):
        state = engine.step(state, dt=1.0)
    burnt_x_m = mesh.cell_positions_xyz[state.has_ignited, 0]
    no_wind_offset = float(burnt_x_m.mean() - mesh.cell_positions_xyz[center, 0])
    assert abs(no_wind_offset) < abs(np.mean(offsets))


def test_engine_satisfies_spread_engine_protocol() -> None:
    assert isinstance(CellularAutomatonSpreadEngine(), SpreadEngineProtocol)


def test_step_before_initialize_raises_a_named_error() -> None:
    engine = CellularAutomatonSpreadEngine()
    state = FireState(
        ignition_times_s=np.full(4, np.inf),
        burnout_times_s=np.full(4, np.inf),
        is_burning=np.zeros(4, dtype=np.bool_),
        has_ignited=np.zeros(4, dtype=np.bool_),
        current_time_s=0.0,
    )
    with pytest.raises(RuntimeError, match="initialize must be called"):
        engine.step(state, dt=1.0)


def test_ignite_cells_marks_only_the_requested_cells() -> None:
    mesh = build_mesh()
    engine = build_engine(0)
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    requested = np.array([0, 5, 9], dtype=np.int64)
    ignited = engine.ignite_cells(state, requested)
    assert np.array_equal(np.flatnonzero(ignited.is_burning), requested)
    assert np.array_equal(np.flatnonzero(ignited.has_ignited), requested)
    assert ignited.current_time_s == state.current_time_s
    np.testing.assert_array_equal(ignited.ignition_times_s[requested], 0.0)
    np.testing.assert_array_equal(ignited.burnout_times_s[requested], BURN_DURATION_S)


def test_ignite_cells_skips_cells_without_fuel() -> None:
    mesh = build_mesh()
    engine = build_engine(0)
    state = engine.initialize(
        mesh,
        UniformScalarField(0.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    ignited = engine.ignite_cells(state, np.array([0], dtype=np.int64))
    assert not ignited.has_ignited.any()


def test_burnt_cell_never_reignites() -> None:
    mesh = build_mesh()
    engine = build_engine(0)
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    center = center_cell_index(mesh)
    state = engine.ignite_cells(state, np.array([center], dtype=np.int64))
    for _ in range(40):
        state = engine.step(state, dt=1.0)
        assert not (
            state.is_burning & (state.burnout_times_s <= state.current_time_s)
        ).any()
    assert state.has_ignited[center]
    assert not state.is_burning[center]


def test_same_seed_reproduces_the_same_run() -> None:
    def run() -> FireState:
        mesh = build_mesh()
        engine = build_engine(7)
        state = engine.initialize(
            mesh,
            UniformScalarField(1.0),
            ConstantWindField.from_speed_and_bearing(WIND_SPEED_M_PER_S, 0.0),
        )
        state = engine.ignite_cells(
            state, np.array([center_cell_index(mesh)], dtype=np.int64)
        )
        for _ in range(10):
            state = engine.step(state, dt=1.0)
        return state

    np.testing.assert_array_equal(run().has_ignited, run().has_ignited)
