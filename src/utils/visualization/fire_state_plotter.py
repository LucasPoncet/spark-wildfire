"""Turns a FireState into pictures. Owns no simulation logic of its own.

May import from anywhere in the repository; nothing imports it back. Every
function here takes data and returns either a plain numpy image or a
`matplotlib.figure.Figure` — never calls `savefig`, that is `scripts/`' job.

Takes a concrete SquareGridMesh rather than a MeshProtocol because a raster
image needs the grid dimensions the protocol deliberately does not expose.
An unstructured mesh needs a different renderer, not this one.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from src.spark.fire.fire_state import FireState
from src.spark.terrain.square_grid_mesh import SquareGridMesh

UNBURNT_GROUND_COLOR_RGB = (42, 38, 32)
BURNT_COLOR_RGB = (43, 38, 34)


def render_fire_state_rgb_image(
    state: FireState,
    mesh: SquareGridMesh,
    fuel_density_fraction: npt.NDArray[np.float64],
    burn_duration_s: float,
    fuel_ignition_threshold_fraction: float,
) -> npt.NDArray[np.uint8]:
    """Render one FireState as an (h, w, 3) uint8 image, independent of any backend.

    Bare ground is a flat grey-brown, unburnt fuel is green tinted by its
    density, a burning cell shifts from bright orange towards red as it
    approaches burnout, and burnt cells are a flat dark grey.

    Args:
        state: The FireState to render.
        mesh: The square grid the state lives on, used to reshape the
            per-cell arrays back into an image.
        fuel_density_fraction: Float64 array of shape (cell_count,), the
            fuel field sampled once at `mesh.cell_positions_xyz`.
        burn_duration_s: The engine's configured burn duration, used to
            colour a burning cell by how far through its burn it is.
        fuel_ignition_threshold_fraction: Fuel density above which a cell is
            drawn as vegetation rather than as bare ground. Supplied by the
            caller so this module stays independent of any one engine.

    Returns:
        Uint8 array of shape (mesh.n_y, mesh.n_x, 3).
    """
    is_flammable = fuel_density_fraction > fuel_ignition_threshold_fraction
    is_burnt = state.has_ignited & ~state.is_burning

    red_channel = np.where(
        is_flammable, 40 + fuel_density_fraction * 30, UNBURNT_GROUND_COLOR_RGB[0]
    )
    green_channel = np.where(
        is_flammable, 70 + fuel_density_fraction * 70, UNBURNT_GROUND_COLOR_RGB[1]
    )
    blue_channel = np.where(
        is_flammable, 38 + fuel_density_fraction * 20, UNBURNT_GROUND_COLOR_RGB[2]
    )

    burn_age_fraction = np.clip(
        (state.current_time_s - state.ignition_times_s) / burn_duration_s, 0.0, 1.0
    )
    red_channel = np.where(state.is_burning, 246 - burn_age_fraction * 40, red_channel)
    green_channel = np.where(
        state.is_burning, 161 - burn_age_fraction * 120, green_channel
    )
    blue_channel = np.where(state.is_burning, 60 - burn_age_fraction * 55, blue_channel)

    red_channel = np.where(is_burnt, BURNT_COLOR_RGB[0], red_channel)
    green_channel = np.where(is_burnt, BURNT_COLOR_RGB[1], green_channel)
    blue_channel = np.where(is_burnt, BURNT_COLOR_RGB[2], blue_channel)

    image = np.stack([red_channel, green_channel, blue_channel], axis=1)
    return image.reshape(mesh.n_y, mesh.n_x, 3).astype(np.uint8)


def plot_fire_state(
    state: FireState,
    mesh: SquareGridMesh,
    fuel_density_fraction: npt.NDArray[np.float64],
    burn_duration_s: float,
    fuel_ignition_threshold_fraction: float,
) -> Figure:
    """Render one FireState as a standalone, axis-labelled figure.

    Args:
        state: The FireState to render.
        mesh: The square grid the state lives on.
        fuel_density_fraction: Float64 array of shape (cell_count,), the
            fuel field sampled once at `mesh.cell_positions_xyz`.
        burn_duration_s: The engine's configured burn duration.
        fuel_ignition_threshold_fraction: Fuel density above which a cell is
            drawn as vegetation rather than as bare ground.

    Returns:
        A matplotlib Figure. Callers save it (`figure.savefig(...)`) or
        display it; this function never does either.
    """
    image = render_fire_state_rgb_image(
        state,
        mesh,
        fuel_density_fraction,
        burn_duration_s,
        fuel_ignition_threshold_fraction,
    )
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.imshow(
        image,
        origin="lower",
        extent=(0.0, mesh.config.extent_x_m, 0.0, mesh.config.extent_y_m),
    )
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title(f"t = {state.current_time_s:.1f} s")
    return figure


def plot_ignition_time_map(
    state: FireState,
    mesh: SquareGridMesh,
    front_masks: list[npt.NDArray[np.bool_]],
    front_mask_times_s: list[float],
) -> Figure:
    """Ignition time as a heatmap, with the front overlaid at chosen instants.

    The heatmap is the whole run at once: contours of equal ignition time are
    the front's successive positions. The overlaid masks pick out a few of
    them so the reader can tie the picture back to the observations.

    Args:
        state: The final FireState, carrying every ignition time.
        mesh: The square grid the state lives on.
        front_masks: Bool arrays of shape (cell_count,), one per instant drawn.
        front_mask_times_s: Simulation time of each mask, in seconds.

    Returns:
        A matplotlib Figure. Callers save it; this function never does.

    Raises:
        ValueError: If the masks and their times do not have the same length.
    """
    if len(front_masks) != len(front_mask_times_s):
        raise ValueError("every front mask needs exactly one time")

    ignition_times_s = np.where(
        np.isfinite(state.ignition_times_s), state.ignition_times_s, np.nan
    ).reshape(mesh.n_y, mesh.n_x)

    figure = Figure(figsize=(3.8, 3.2))
    axes = figure.add_subplot(111)
    image = axes.imshow(
        ignition_times_s,
        origin="lower",
        extent=(0.0, mesh.config.extent_x_m, 0.0, mesh.config.extent_y_m),
        cmap="magma",
    )
    figure.colorbar(image, ax=axes, label="ignition time (s)")

    for front_mask, mask_time_s in zip(front_masks, front_mask_times_s, strict=True):
        front_positions_xy_m = mesh.cell_positions_xyz[front_mask, :2]
        if front_positions_xy_m.shape[0] == 0:
            continue
        axes.scatter(
            front_positions_xy_m[:, 0],
            front_positions_xy_m[:, 1],
            s=4,
            label=f"t = {mask_time_s:.0f} s",
        )

    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title("Front evolution")
    if front_masks:
        axes.legend(fontsize="xx-small", loc="upper right", framealpha=0.85)
    figure.tight_layout()
    return figure
