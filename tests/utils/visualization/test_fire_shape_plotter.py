import numpy as np
import pytest
from matplotlib.figure import Figure

from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.cellular_automaton_spread_engine import (
    FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
    CellularAutomatonSpreadEngine,
    CellularAutomatonSpreadEngineConfig,
)
from src.spark.fire.fire_state import FireState
from src.spark.inverse.steered_response_power import build_candidate_grid_xyz
from src.spark.terrain.square_grid_mesh import SquareGridMesh
from src.utils.visualization.fire_shape_plotter import (
    plot_fire_and_steered_response_power,
)

DOMAIN_EXTENT_M: float = 20.0
MAP_SPACING_M: float = 2.0
BURN_DURATION_S: float = 20.0
SOURCE_HEIGHT_M: float = 0.5
FRONT_CENTRE_XY_M = np.array([8.0, 12.0])
RECEIVERS_XYZ_M = np.array(
    [[2.0, 10.0, 1.5], [18.0, 10.0, 1.5], [10.0, 18.0, 1.5], [10.0, 2.0, 1.5]]
)


def build_fire() -> tuple[SquareGridMesh, FireState, np.ndarray]:
    mesh = SquareGridMesh.from_values(DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, 1.0, False)
    engine = CellularAutomatonSpreadEngine(
        CellularAutomatonSpreadEngineConfig(
            random_seed=0, burn_duration_s=BURN_DURATION_S
        )
    )
    fuel_field = UniformScalarField(1.0)
    state = engine.initialize(
        mesh, fuel_field, ConstantWindField.from_speed_and_bearing(0.0, 0.0)
    )
    state = engine.ignite_cells(state, np.array([0], dtype=np.int64))
    return mesh, state, fuel_field.sample(mesh.cell_positions_xyz)


def build_map() -> tuple[np.ndarray, np.ndarray]:
    grid = build_candidate_grid_xyz(
        DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, MAP_SPACING_M, SOURCE_HEIGHT_M
    )
    distances_m = np.linalg.norm(
        grid.positions_xyz_m[:, :2] - FRONT_CENTRE_XY_M, axis=1
    )
    return grid.positions_xyz_m, np.exp(-0.5 * (distances_m / 3.0) ** 2)


def build_frame(
    estimated_positions_xy_m: np.ndarray, covariances_m2: list[np.ndarray]
) -> Figure:
    mesh, state, fuel_density_fraction = build_fire()
    candidate_positions_xyz_m, power_map = build_map()
    return plot_fire_and_steered_response_power(
        state,
        mesh,
        fuel_density_fraction,
        BURN_DURATION_S,
        FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
        power_map,
        candidate_positions_xyz_m,
        RECEIVERS_XYZ_M,
        FRONT_CENTRE_XY_M.reshape(1, 2),
        estimated_positions_xy_m,
        covariances_m2,
        "test frame",
    )


def test_the_frame_carries_one_panel_for_the_fire_and_one_for_the_map() -> None:
    figure = build_frame(np.array([[8.5, 11.5]]), [np.diag([1.0, 4.0])])
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert figure.axes[0].get_title().startswith("fire at t =")
    assert figure.axes[1].get_title().startswith("steered response power")


def test_both_panels_share_the_domain_so_the_shapes_can_be_compared() -> None:
    figure = build_frame(np.array([[8.5, 11.5]]), [np.diag([1.0, 4.0])])
    fire_axes, map_axes = figure.axes
    assert fire_axes.get_xlim() == pytest.approx(map_axes.get_xlim(), abs=MAP_SPACING_M)
    assert fire_axes.get_ylim() == pytest.approx(map_axes.get_ylim(), abs=MAP_SPACING_M)
    assert fire_axes.get_aspect() == 1.0
    assert map_axes.get_aspect() == 1.0


def test_the_frame_survives_a_window_in_which_nothing_was_located() -> None:
    figure = build_frame(np.empty((0, 2)), [])
    assert isinstance(figure, Figure)
    assert "0 accepted" in figure.axes[1].get_title()


def test_one_accepted_source_is_named_in_the_singular() -> None:
    figure = build_frame(np.array([[8.5, 11.5]]), [np.diag([1.0, 1.0])])
    assert "1 accepted (source)" in figure.axes[1].get_title()


def test_every_estimate_gets_its_own_error_ellipse() -> None:
    estimates = np.array([[8.5, 11.5], [12.0, 6.0], [4.0, 15.0]])
    figure = build_frame(estimates, [np.diag([1.0, 4.0])] * estimates.shape[0])
    assert len(figure.axes[1].patches) == estimates.shape[0]
