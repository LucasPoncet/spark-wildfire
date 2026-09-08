"""Runs the cellular automaton fire simulation and displays it live.

Usage:
    python scripts/run_fire_simulation.py

A window opens showing the grid. Click anywhere on it to start a fire at
that cell; the simulation keeps advancing on its own from the moment the
window opens.

Zero domain logic lives here: this script only builds the day-one
components (a flat square grid, uniform fuel, constant wind), wires them
into the cellular automaton engine, and drives the animation loop. Swap
any of `fuel_field`, `wind_field`, or `engine` below for a different
implementation satisfying the same Protocol and nothing else changes.
"""

import sys
from pathlib import Path
import numpy as np

# Lets this script run directly, from any working directory, without an
# editable install: `src/` holds `spark` and `utils` as top-level packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseEvent

from spark.fields.constant_wind_field import ConstantWindField
from spark.fields.random_tree_placement_field import RandomTreePlacementField
from spark.fields.patchy_density_field import PatchyDensityField
from spark.fire.cellular_automaton_spread_engine import (
    CellularAutomatonSpreadEngine,
    CellularAutomatonSpreadEngineConfig,
)
from spark.terrain.square_grid_mesh import SquareGridMesh, SquareGridMeshConfig
from utils.visualization.fire_state_plotter import render_fire_state_rgb_image

GRID_EXTENT_M = 100.0
CELL_SPACING_M = 0.5
TIME_STEP_S = 1.0
ANIMATION_FRAME_INTERVAL_MS = 150
WIND_SPEED_M_PER_S = 3.0
WIND_BEARING_RAD = 0.0
TREE_DENSITY_PER_M2 = 0.2
TREE_FUEL_LOAD = 0.25
TREE_INFLUENCE_RADIUS_M = 2.5
TREE_LAYOUT_SEED = 42
BURN_DURATION_S = 1

#density
TREE_BACKGROUND_DENSITY_PER_M2 = 0.01
TREE_PATCH_CENTERS_XY = np.array([[25.0, 25.0], [70.0, 60.0], [40.0, 80.0]])
TREE_PATCH_RADIUS_M = 15.0

def main() -> None:
    """Build the day-one simulation and show it in an interactive window."""
    mesh = SquareGridMesh(
        SquareGridMeshConfig(
            extent_x_m=GRID_EXTENT_M,
            extent_y_m=GRID_EXTENT_M,
            cell_spacing_m=CELL_SPACING_M,
            use_diagonal_neighbors=True,
        )
    )
    density_field = PatchyDensityField(
        patch_centers_xy=TREE_PATCH_CENTERS_XY,
        patch_peak_density_per_m2=TREE_DENSITY_PER_M2,
        patch_radius_m=TREE_PATCH_RADIUS_M,
        background_density_per_m2=TREE_BACKGROUND_DENSITY_PER_M2,
    )
    fuel_field = RandomTreePlacementField(
        extent_x_m=GRID_EXTENT_M,
        extent_y_m=GRID_EXTENT_M,
        tree_density_per_m2=TREE_DENSITY_PER_M2,
        tree_fuel_load=TREE_FUEL_LOAD,
        influence_radius_m=TREE_INFLUENCE_RADIUS_M,
        seed=TREE_LAYOUT_SEED,
        density_field=density_field,
    )
    wind_field = ConstantWindField.from_speed_and_bearing(WIND_SPEED_M_PER_S, WIND_BEARING_RAD)
    engine = CellularAutomatonSpreadEngine(
        CellularAutomatonSpreadEngineConfig(burn_duration_s=BURN_DURATION_S)
    )
    fuel_density_fraction = fuel_field.sample(mesh.cell_positions_xyz)

    # A dict, not a bare variable, so the nested callbacks below can rebind
    # it — matplotlib's animation and event callbacks can only close over
    # a mutable container, not reassign an outer local.
    simulation = {"state": engine.initialize(mesh, fuel_field, wind_field)}
    simulation["state"], _ = engine.ignite_cell(simulation["state"], mesh.cell_count // 2)

    figure, axes = plt.subplots()
    image_artist = axes.imshow(
        render_fire_state_rgb_image(
            simulation["state"], mesh, fuel_density_fraction, BURN_DURATION_S
        ),
        origin="lower",
        extent=(0.0, GRID_EXTENT_M, 0.0, GRID_EXTENT_M),
    )
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    title_artist = axes.set_title("t = 0.0 s — click to ignite a cell")

    def ignite_cell_at_click(event: MouseEvent) -> None:
        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            return
        column_index = round(event.xdata / CELL_SPACING_M)
        row_index = round(event.ydata / CELL_SPACING_M)
        if 0 <= column_index < mesh.n_x and 0 <= row_index < mesh.n_y:
            cell_index = row_index * mesh.n_x + column_index
            simulation["state"], _ = engine.ignite_cell(simulation["state"], cell_index)

    def advance_one_frame(_frame_number: int):
        simulation["state"] = engine.step(simulation["state"], dt=TIME_STEP_S)
        image_artist.set_data(
            render_fire_state_rgb_image(
                simulation["state"], mesh, fuel_density_fraction, BURN_DURATION_S
            )
        )
        title_artist.set_text(
            f"t = {simulation['state'].current_time_s:.1f} s — click to ignite a cell"
        )
        return image_artist, title_artist

    figure.canvas.mpl_connect("button_press_event", ignite_cell_at_click)
    # Must stay referenced for the animation to keep running once `main`
    # blocks on `plt.show()` below — matplotlib does not hold this itself.
    animation = FuncAnimation(
        figure, advance_one_frame, interval=ANIMATION_FRAME_INTERVAL_MS, cache_frame_data=False
    )
    plt.show()
    del animation  # keeps linters from flagging an "unused" variable


if __name__ == "__main__":
    main()
