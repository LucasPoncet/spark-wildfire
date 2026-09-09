"""Runs a fire simulation and displays it live.

Usage:
    uv run python scripts/run_fire_simulation.py
    uv run python scripts/run_fire_simulation.py --configs configs/e3
    uv run python scripts/run_fire_simulation.py --time-step-s 5

A window opens showing the grid. Click anywhere on it to start a fire at
that cell; the simulation keeps advancing on its own from the moment the
window opens.

Zero domain logic lives here. The mesh, engine and wind come from
`simulation_context_factory`, so `--configs` swaps the engine the same way it
does for the rendering script. The patchy tree fuel is built here rather than
by the factory because it is a display asset: it makes the front visibly
wander, and no configuration section describes it yet.

The mesh is narrowed back to a concrete `SquareGridMesh` before display,
because a raster image needs the row and column structure `MeshProtocol`
deliberately does not expose. Narrowing is not construction: the composition
root is still the only place a mesh is built.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseEvent

from src.config.component_registry import RATE_OF_SPREAD_ENGINE
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    ForwardSimulationConfiguration,
    load_forward_simulation_configuration,
)
from src.config.simulation_context import SimulationContext
from src.config.simulation_context_factory import build_simulation_context
from src.spark.fields.patchy_density_field import PatchyDensityField
from src.spark.fields.random_tree_placement_field import RandomTreePlacementField
from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fire.cellular_automaton_spread_engine import (
    FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
)
from src.spark.terrain.square_grid_mesh import SquareGridMesh
from src.utils.visualization.fire_state_plotter import render_fire_state_rgb_image

ANIMATION_FRAME_INTERVAL_MS = 150
TREE_DENSITY_PER_M2 = 0.1
TREE_FUEL_LOAD_KG_PER_M2 = 0.25
TREE_INFLUENCE_RADIUS_M = 2.5
TREE_LAYOUT_SEED = 100
CELLULAR_AUTOMATON_BURN_DURATION_S = 10.0
RATE_OF_SPREAD_DEFAULT_TIME_STEP_S = 10.0
CELLULAR_AUTOMATON_DEFAULT_TIME_STEP_S = 1.0

TREE_BACKGROUND_DENSITY_PER_M2 = 0.01
TREE_PATCH_CENTERS_FRACTION_XY = np.array(
    [[0.25, 0.25], [0.70, 0.60], [0.40, 0.80]], dtype=np.float64
)
TREE_PATCH_RADIUS_FRACTION = 0.15


def parse_arguments() -> argparse.Namespace:
    """Read the configuration directory and timestep from the command line.

    Returns:
        Parsed arguments carrying `configs` and `time_step_s`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    parser.add_argument(
        "--time-step-s",
        type=float,
        default=None,
        help="Simulated seconds per frame. Defaults to 1 s for the cellular "
        "automaton and 10 s for the rate of spread engine, whose front "
        "creeps at millimetres per second against the wind.",
    )
    return parser.parse_args()


def build_tree_fuel_field(extent_x_m: float, extent_y_m: float) -> ScalarFieldProtocol:
    """Build the patchy tree fuel the demo displays.

    Args:
        extent_x_m: Domain extent along x, in metres.
        extent_y_m: Domain extent along y, in metres.

    Returns:
        The fuel field, behind its protocol.
    """
    density_field = PatchyDensityField(
        patch_centers_xy=TREE_PATCH_CENTERS_FRACTION_XY
        * np.array([extent_x_m, extent_y_m]),
        patch_peak_density_per_m2=TREE_DENSITY_PER_M2,
        patch_radius_m=TREE_PATCH_RADIUS_FRACTION * extent_x_m,
        background_density_per_m2=TREE_BACKGROUND_DENSITY_PER_M2,
    )
    field: ScalarFieldProtocol = RandomTreePlacementField(
        extent_x_m=extent_x_m,
        extent_y_m=extent_y_m,
        tree_density_per_m2=TREE_DENSITY_PER_M2,
        tree_fuel_load_kg_per_m2=TREE_FUEL_LOAD_KG_PER_M2,
        influence_radius_m=TREE_INFLUENCE_RADIUS_M,
        seed=TREE_LAYOUT_SEED,
        density_field=density_field,
    )
    return field


def resolve_display_settings(
    config: ForwardSimulationConfiguration,
    context: SimulationContext,
    requested_time_step_s: float | None,
) -> tuple[float, float]:
    """Pick how long a cell burns and how far each frame advances.

    Args:
        config: Run configuration, naming the engine.
        context: The built context, carrying the fuel bed.
        requested_time_step_s: Timestep from the command line, or None.

    Returns:
        The burn duration used only for shading, and the frame timestep,
        both in seconds.
    """
    if config.fire.spread_engine_name == RATE_OF_SPREAD_ENGINE:
        burn_duration_s = context.fuel.residence_time_s
        default_time_step_s = RATE_OF_SPREAD_DEFAULT_TIME_STEP_S
    else:
        burn_duration_s = CELLULAR_AUTOMATON_BURN_DURATION_S
        default_time_step_s = CELLULAR_AUTOMATON_DEFAULT_TIME_STEP_S
    return burn_duration_s, (
        default_time_step_s if requested_time_step_s is None else requested_time_step_s
    )


def main() -> None:
    """Build the simulation from a configuration directory and show it."""
    arguments = parse_arguments()
    config = load_forward_simulation_configuration(arguments.configs)
    context = build_simulation_context(config)
    burn_duration_s, time_step_s = resolve_display_settings(
        config, context, arguments.time_step_s
    )

    mesh = context.mesh
    if not isinstance(mesh, SquareGridMesh):
        raise TypeError(
            "the live display renders a raster image and needs a square grid; "
            f"{type(mesh).__name__} has no row and column structure"
        )
    engine = context.spread_engine
    fuel_field = build_tree_fuel_field(config.mesh.extent_x_m, config.mesh.extent_y_m)
    fuel_density_fraction = np.clip(
        fuel_field.sample(mesh.cell_positions_xyz), 0.0, 1.0
    )
    cells_per_row = round(config.mesh.extent_x_m / config.mesh.cell_spacing_m) + 1

    simulation = {"state": engine.initialize(mesh, fuel_field, context.wind_field)}
    simulation["state"] = engine.ignite_cells(
        simulation["state"], context.ignition_cell_indices
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
        extent=(0.0, config.mesh.extent_x_m, 0.0, config.mesh.extent_y_m),
    )
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    title_artist = axes.set_title(
        f"{config.fire.spread_engine_name} — t = 0.0 s — click to ignite"
    )

    def ignite_cell_at_click(event: MouseEvent) -> None:
        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            return
        column_index = round(event.xdata / config.mesh.cell_spacing_m)
        row_index = round(event.ydata / config.mesh.cell_spacing_m)
        cell_index = row_index * cells_per_row + column_index
        if 0 <= cell_index < mesh.cell_count:
            simulation["state"] = engine.ignite_cells(
                simulation["state"], np.array([cell_index], dtype=np.int64)
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
            f"{config.fire.spread_engine_name} — "
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
