"""The fire beside what the array made of it, as one two-panel frame.

Left is the ground truth the estimator never sees; right is the steered
response power map it built from twenty receiver waveforms alone. Putting them
on shared axes is the whole point: a shape estimate is judged by where its ridge
sits relative to the burning contour, which a map on its own cannot show.

Composed from the two plotters that already own these panels —
`fire_state_plotter.render_fire_state_rgb_image` for the raster and
`steered_response_power_plotter.reshape_map_to_grid` and `add_error_ellipse`
for the map — so a change to either shows up here rather than drifting from it.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.spark.fire.fire_state import FireState
from src.spark.terrain.square_grid_mesh import SquareGridMesh
from src.utils.array_types import Float64Array
from src.utils.visualization.fire_state_plotter import render_fire_state_rgb_image
from src.utils.visualization.steered_response_power_plotter import (
    add_error_ellipse,
    reshape_map_to_grid,
)

MAP_COLORMAP: str = "magma"
RECEIVER_COLOR: str = "#2f5d8a"
FRONT_COLOR: str = "#e08a1e"
ESTIMATE_COLOR: str = "#43d6c4"
FIGURE_SIZE_INCHES: tuple[float, float] = (11.0, 4.8)


def draw_receivers(axes: Axes, receiver_positions_xyz_m: Float64Array) -> None:
    """Marks every receiver on one panel.

    Args:
        axes: Axes to draw on.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
    """
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    axes.scatter(
        receivers[:, 0],
        receivers[:, 1],
        marker="v",
        s=26,
        color=RECEIVER_COLOR,
        edgecolors="white",
        linewidths=0.4,
        label="receivers",
        zorder=4,
    )


def draw_fire_panel(
    axes: Axes,
    state: FireState,
    mesh: SquareGridMesh,
    fuel_density_fraction: Float64Array,
    burn_duration_s: float,
    fuel_ignition_threshold_fraction: float,
    receiver_positions_xyz_m: Float64Array,
) -> None:
    """Draws the true fire state as a raster with the array around it.

    Args:
        axes: Axes to draw on.
        state: The fire state at this observation.
        mesh: The square grid the state lives on.
        fuel_density_fraction: Fuel field sampled at every cell, shape
            `(cell_count,)`.
        burn_duration_s: How long a cell burns, used only to shade it.
        fuel_ignition_threshold_fraction: Fuel density above which a cell is
            drawn as vegetation rather than bare ground.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
    """
    axes.imshow(
        render_fire_state_rgb_image(
            state,
            mesh,
            np.asarray(fuel_density_fraction, dtype=np.float64),
            burn_duration_s,
            fuel_ignition_threshold_fraction,
        ),
        origin="lower",
        extent=(0.0, mesh.config.extent_x_m, 0.0, mesh.config.extent_y_m),
    )
    draw_receivers(axes, receiver_positions_xyz_m)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title(f"fire at t = {state.current_time_s:.0f} s")


def draw_map_panel(
    axes: Axes,
    steered_response_power_map: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    front_positions_xy_m: Float64Array,
    estimated_positions_xy_m: Float64Array,
    estimated_covariances_m2: list[Float64Array],
) -> None:
    """Draws the map with the burning contour and the accepted sources on it.

    The true front is drawn here as well as on the left panel, because the
    question this figure exists to answer is whether the map's ridge sits on
    the contour, which needs both in one set of axes.

    Args:
        axes: Axes to draw on.
        steered_response_power_map: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres of that map, shape
            `(n_cells, 3)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        front_positions_xy_m: Burning contour, shape `(n_front_cells, 2)`.
        estimated_positions_xy_m: Accepted sources, shape `(n_sources, 2)`.
        estimated_covariances_m2: One two-by-two covariance per accepted source.
    """
    x_m, y_m, map_grid = reshape_map_to_grid(
        steered_response_power_map, candidate_positions_xyz_m
    )
    axes.pcolormesh(x_m, y_m, map_grid, cmap=MAP_COLORMAP, shading="nearest")

    front = np.atleast_2d(np.asarray(front_positions_xy_m, dtype=np.float64))
    if front.size:
        axes.scatter(
            front[:, 0],
            front[:, 1],
            marker=".",
            s=6,
            color=FRONT_COLOR,
            alpha=0.85,
            label="true front",
            zorder=3,
        )

    estimates = np.atleast_2d(np.asarray(estimated_positions_xy_m, dtype=np.float64))
    if estimates.size:
        axes.scatter(
            estimates[:, 0],
            estimates[:, 1],
            marker="x",
            s=52,
            color=ESTIMATE_COLOR,
            linewidths=1.6,
            label="estimated sources",
            zorder=5,
        )
        for position_xy_m, covariance_m2 in zip(
            estimates, estimated_covariances_m2, strict=False
        ):
            add_error_ellipse(axes, position_xy_m, covariance_m2, ESTIMATE_COLOR)

    draw_receivers(axes, receiver_positions_xyz_m)
    axes.set_xlim(float(x_m.min()), float(x_m.max()))
    axes.set_ylim(float(y_m.min()), float(y_m.max()))
    axes.set_xlabel("x (m)")
    axes.set_title(
        f"steered response power, {estimates.shape[0]} accepted "
        f"({'source' if estimates.shape[0] == 1 else 'sources'})"
    )


def plot_fire_and_steered_response_power(
    state: FireState,
    mesh: SquareGridMesh,
    fuel_density_fraction: Float64Array,
    burn_duration_s: float,
    fuel_ignition_threshold_fraction: float,
    steered_response_power_map: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    front_positions_xy_m: Float64Array,
    estimated_positions_xy_m: Float64Array,
    estimated_covariances_m2: list[Float64Array],
    title: str,
) -> Figure:
    """Builds one frame: the fire on the left, what the array heard on the right.

    Args:
        state: The fire state at this observation.
        mesh: The square grid the state lives on.
        fuel_density_fraction: Fuel field sampled at every cell, shape
            `(cell_count,)`.
        burn_duration_s: How long a cell burns, used only to shade it.
        fuel_ignition_threshold_fraction: Fuel density above which a cell is
            drawn as vegetation rather than bare ground.
        steered_response_power_map: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres of that map, shape
            `(n_cells, 3)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        front_positions_xy_m: Burning contour, shape `(n_front_cells, 2)`.
        estimated_positions_xy_m: Accepted sources, shape `(n_sources, 2)`.
        estimated_covariances_m2: One two-by-two covariance per accepted source.
        title: Caption for the whole frame.

    Returns:
        A matplotlib Figure. Callers save it; this function never does.
    """
    figure = Figure(figsize=FIGURE_SIZE_INCHES)
    fire_axes = figure.add_subplot(121)
    map_axes = figure.add_subplot(122)

    draw_fire_panel(
        fire_axes,
        state,
        mesh,
        fuel_density_fraction,
        burn_duration_s,
        fuel_ignition_threshold_fraction,
        receiver_positions_xyz_m,
    )
    draw_map_panel(
        map_axes,
        steered_response_power_map,
        candidate_positions_xyz_m,
        receiver_positions_xyz_m,
        front_positions_xy_m,
        estimated_positions_xy_m,
        estimated_covariances_m2,
    )
    for axes in (fire_axes, map_axes):
        axes.set_aspect("equal")
    map_axes.legend(loc="upper right", fontsize=7, framealpha=0.7)
    figure.suptitle(title)
    figure.tight_layout()
    return figure
