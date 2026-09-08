"""Runs a fire simulation and displays it live.

Usage:
    uv run python scripts/run_fire_simulation.py
    uv run python scripts/run_fire_simulation.py --engine rate-of-spread
    uv run python scripts/run_fire_simulation.py --engine rate-of-spread --time-step-s 5

A window opens showing the grid. Click anywhere on it to start a fire at
that cell; the simulation keeps advancing on its own from the moment the
window opens.

Zero domain logic lives here: this script builds the day-one components (a
flat square grid, a patchy random forest, constant wind), wires them into
whichever engine was asked for, and drives the animation loop. Both engines
are held as a `SpreadEngineProtocol`, so everything below `build_engine` is
identical for either one — that is the protocol earning its keep. When
`simulation_context_factory.py` lands, `build_engine` moves there and this
file keeps only the animation.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseEvent

from spark.fields.constant_wind_field import ConstantWindField
from spark.fields.patchy_density_field import PatchyDensityField
from spark.fields.random_tree_placement_field import RandomTreePlacementField
from spark.fire.cellular_automaton_spread_engine import (
    FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
    CellularAutomatonSpreadEngine,
    CellularAutomatonSpreadEngineConfig,
)
from spark.fire.fuel_properties import FuelProperties
from spark.fire.rate_of_spread_engine import RateOfSpreadEngine
from spark.fire.spread_engine_protocol import SpreadEngineProtocol
from spark.terrain.square_grid_mesh import SquareGridMesh, SquareGridMeshConfig
from utils.visualization.fire_state_plotter import render_fire_state_rgb_image

CELLULAR_AUTOMATON_ENGINE = "cellular-automaton"
RATE_OF_SPREAD_ENGINE = "rate-of-spread"

GRID_EXTENT_M = 100.0
CELL_SPACING_M = 0.5
ANIMATION_FRAME_INTERVAL_MS = 150
WIND_SPEED_M_PER_S = 3.0
WIND_BEARING_RAD = 0.0
TREE_DENSITY_PER_M2 = 0.1
TREE_FUEL_LOAD_KG_PER_M2 = 0.25
TREE_INFLUENCE_RADIUS_M = 2.5
TREE_LAYOUT_SEED = 42
CELLULAR_AUTOMATON_BURN_DURATION_S = 10.0

TREE_BACKGROUND_DENSITY_PER_M2 = 0.01
TREE_PATCH_CENTERS_FRACTION_XY = np.array(
    [[0.25, 0.25], [0.70, 0.60], [0.40, 0.80]], dtype=np.float64
)
TREE_PATCH_RADIUS_FRACTION = 0.15


def parse_arguments() -> argparse.Namespace:
    """Read the engine choice and timestep from the command line.

    Returns:
        Parsed arguments carrying `engine` and `time_step_s`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine",
        choices=[CELLULAR_AUTOMATON_ENGINE, RATE_OF_SPREAD_ENGINE],
        default=CELLULAR_AUTOMATON_ENGINE,
        help="Which spread engine to run. Default: %(default)s.",
    )
    parser.add_argument(
        "--time-step-s",
        type=float,
        default=None,
        help="Simulated seconds per frame. Defaults to 1 s for the cellular "
        "automaton and 10 s for the rate of spread engine, whose front "
        "creeps at millimetres per second against the wind.",
    )
    return parser.parse_args()


def build_engine(engine_name: str) -> tuple[SpreadEngineProtocol, float, float]:
    """Build the requested engine and the settings that depend on it.

    Args:
        engine_name: One of the two engine identifiers.

    Returns:
        A tuple of the engine, how long one cell burns in seconds, and the
        default timestep in seconds. The burn duration is only used to shade
        burning cells by how far through their burn they are.
    """
    if engine_name == RATE_OF_SPREAD_ENGINE:
        fuel = FuelProperties.pine_needle_litter()
        return RateOfSpreadEngine(fuel), fuel.residence_time_s, 10.0
    return (
        CellularAutomatonSpreadEngine(
            CellularAutomatonSpreadEngineConfig(
                burn_duration_s=CELLULAR_AUTOMATON_BURN_DURATION_S
            )
        ),
        CELLULAR_AUTOMATON_BURN_DURATION_S,
        1.0,
    )


def main() -> None:
    """Build the simulation from the command line and show it in a window."""
    arguments = parse_arguments()
    engine, burn_duration_s, default_time_step_s = build_engine(arguments.engine)
    time_step_s = (
        default_time_step_s if arguments.time_step_s is None else arguments.time_step_s
    )

    mesh = SquareGridMesh(
        SquareGridMeshConfig(
            extent_x_m=GRID_EXTENT_M,
            extent_y_m=GRID_EXTENT_M,
            cell_spacing_m=CELL_SPACING_M,
            use_diagonal_neighbors=True,
        )
    )
    density_field = PatchyDensityField(
        patch_centers_xy=TREE_PATCH_CENTERS_FRACTION_XY * GRID_EXTENT_M,
        patch_peak_density_per_m2=TREE_DENSITY_PER_M2,
        patch_radius_m=TREE_PATCH_RADIUS_FRACTION * GRID_EXTENT_M,
        background_density_per_m2=TREE_BACKGROUND_DENSITY_PER_M2,
    )
    fuel_field = RandomTreePlacementField(
        extent_x_m=GRID_EXTENT_M,
        extent_y_m=GRID_EXTENT_M,
        tree_density_per_m2=TREE_DENSITY_PER_M2,
        tree_fuel_load_kg_per_m2=TREE_FUEL_LOAD_KG_PER_M2,
        influence_radius_m=TREE_INFLUENCE_RADIUS_M,
        seed=TREE_LAYOUT_SEED,
        density_field=density_field,
    )
    wind_field = ConstantWindField.from_speed_and_bearing(
        WIND_SPEED_M_PER_S, WIND_BEARING_RAD
    )
    fuel_density_fraction = np.clip(
        fuel_field.sample(mesh.cell_positions_xyz), 0.0, 1.0
    )

    simulation = {"state": engine.initialize(mesh, fuel_field, wind_field)}
    simulation["state"] = engine.ignite_cells(
        simulation["state"], np.array([mesh.cell_count // 2], dtype=np.int64)
    )

    figure, axes = plt.subplots()
    image_artist = axes.imshow(
        render_fire_state_rgb_image(
            simulation["state"],
            mesh,
            fuel_density_fraction,
            burn_duration_s,
            FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
        ),
        origin="lower",
        extent=(0.0, GRID_EXTENT_M, 0.0, GRID_EXTENT_M),
    )
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    title_artist = axes.set_title(f"{arguments.engine} — t = 0.0 s — click to ignite")

    def ignite_cell_at_click(event: MouseEvent) -> None:
        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            return
        column_index = round(event.xdata / CELL_SPACING_M)
        row_index = round(event.ydata / CELL_SPACING_M)
        if 0 <= column_index < mesh.n_x and 0 <= row_index < mesh.n_y:
            simulation["state"] = engine.ignite_cells(
                simulation["state"],
                np.array([row_index * mesh.n_x + column_index], dtype=np.int64),
            )

    def advance_one_frame(_frame_number: int) -> tuple[Artist, ...]:
        simulation["state"] = engine.step(simulation["state"], dt=time_step_s)
        image_artist.set_data(
            render_fire_state_rgb_image(
                simulation["state"],
                mesh,
                fuel_density_fraction,
                burn_duration_s,
                FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
            )
        )
        title_artist.set_text(
            f"{arguments.engine} — "
            f"t = {simulation['state'].current_time_s:.1f} s — click to ignite"
        )
        return image_artist, title_artist

    figure.canvas.mpl_connect("button_press_event", ignite_cell_at_click)
    _animation = FuncAnimation(
        figure,
        advance_one_frame,
        interval=ANIMATION_FRAME_INTERVAL_MS,
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
