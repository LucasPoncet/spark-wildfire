import numpy as np
from matplotlib.figure import Figure

from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.cellular_automaton_spread_engine import (
    FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
    CellularAutomatonSpreadEngine,
    CellularAutomatonSpreadEngineConfig,
)
from src.spark.terrain.square_grid_mesh import SquareGridMesh
from src.utils.visualization.fire_state_plotter import (
    BURNT_COLOR_RGB,
    plot_fire_state,
    render_fire_state_rgb_image,
)

BURN_DURATION_S = 20.0


def build_scene() -> tuple[SquareGridMesh, CellularAutomatonSpreadEngine, np.ndarray]:
    mesh = SquareGridMesh.from_values(10.0, 10.0, 1.0, False)
    engine = CellularAutomatonSpreadEngine(
        CellularAutomatonSpreadEngineConfig(
            random_seed=0, burn_duration_s=BURN_DURATION_S
        )
    )
    fuel_density_fraction = UniformScalarField(1.0).sample(mesh.cell_positions_xyz)
    return mesh, engine, fuel_density_fraction


def test_rendered_image_has_grid_shape_and_uint8_dtype() -> None:
    mesh, engine, fuel = build_scene()
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    image = render_fire_state_rgb_image(
        state, mesh, fuel, BURN_DURATION_S, FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION
    )
    assert image.shape == (mesh.n_y, mesh.n_x, 3)
    assert image.dtype == np.uint8


def test_burnt_cells_render_in_the_burnt_colour() -> None:
    mesh, engine, fuel = build_scene()
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    state = engine.ignite_cells(state, np.array([0], dtype=np.int64))
    for _ in range(int(BURN_DURATION_S) + 1):
        state = engine.step(state, dt=1.0)
    image = render_fire_state_rgb_image(
        state, mesh, fuel, BURN_DURATION_S, FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION
    )
    assert tuple(image[0, 0]) == BURNT_COLOR_RGB


def test_plot_fire_state_returns_a_figure_without_saving() -> None:
    mesh, engine, fuel = build_scene()
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    figure = plot_fire_state(
        state, mesh, fuel, BURN_DURATION_S, FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION
    )
    assert isinstance(figure, Figure)
