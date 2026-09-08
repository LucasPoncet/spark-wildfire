import numpy as np

from spark.acoustic.burning_cell_source_model import BurningCellSourceModel
from spark.fields.constant_wind_field import ConstantWindField
from spark.fields.uniform_scalar_field import UniformScalarField
from spark.fire.fuel_properties import FuelProperties
from spark.fire.rate_of_spread_engine import RateOfSpreadEngine
from spark.terrain.square_grid_mesh import SquareGridMesh

IGNITED_CELLS = np.array([0, 7, 40], dtype=np.int64)
SOURCE_AMPLITUDE = 2.5


def build_scene() -> tuple[SquareGridMesh, RateOfSpreadEngine]:
    mesh = SquareGridMesh.from_values(10.0, 10.0, 1.0, False)
    engine = RateOfSpreadEngine(FuelProperties.pine_needle_litter())
    return mesh, engine


def test_three_ignited_cells_give_three_sources() -> None:
    mesh, engine = build_scene()
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    state = engine.ignite_cells(state, IGNITED_CELLS)
    positions_xyz, amplitudes = BurningCellSourceModel(
        mesh, SOURCE_AMPLITUDE
    ).compute_sources(state)
    assert positions_xyz.shape == (3, 3)
    assert amplitudes.shape == (3,)


def test_source_positions_match_the_burning_cells() -> None:
    mesh, engine = build_scene()
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    state = engine.ignite_cells(state, IGNITED_CELLS)
    positions_xyz, _ = BurningCellSourceModel(mesh).compute_sources(state)
    np.testing.assert_array_equal(positions_xyz, mesh.cell_positions_xyz[IGNITED_CELLS])


def test_every_source_carries_the_configured_amplitude() -> None:
    mesh, engine = build_scene()
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    state = engine.ignite_cells(state, IGNITED_CELLS)
    _, amplitudes = BurningCellSourceModel(mesh, SOURCE_AMPLITUDE).compute_sources(
        state
    )
    np.testing.assert_array_equal(amplitudes, SOURCE_AMPLITUDE)


def test_nothing_burning_gives_empty_arrays() -> None:
    mesh, engine = build_scene()
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    positions_xyz, amplitudes = BurningCellSourceModel(mesh).compute_sources(state)
    assert positions_xyz.shape == (0, 3)
    assert amplitudes.shape == (0,)


def test_burnt_out_cells_stop_being_sources() -> None:
    mesh, engine = build_scene()
    fuel = FuelProperties.pine_needle_litter()
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    state = engine.ignite_cells(state, np.array([0], dtype=np.int64))
    model = BurningCellSourceModel(mesh)
    assert model.compute_sources(state)[0].shape == (1, 3)
    for _ in range(int(fuel.residence_time_s) + 1):
        state = engine.step(state, dt=1.0)
    assert state.has_ignited[0]
    assert model.compute_sources(state)[0].shape == (0, 3)
