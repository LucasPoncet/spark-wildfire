"""Scene geometry as an identifiability diagram, not just a map.

Takes data and returns a `matplotlib.figure.Figure`. Never calls `savefig`;
saving belongs to `scripts/`. Imports nothing from `inverse/` or `fire/`: the
near-singular band is pure geometry, `|r1 / r2 - 1| < tolerance`, so it can be
drawn without asking the estimator anything.
"""

import numpy as np
from matplotlib.figure import Figure

from src.utils.array_types import BoolArray, Float64Array

BAND_GRID_RESOLUTION: int = 240
RANGE_RATIO_FLOOR_M: float = 1e-9


def compute_range_ratio_field(
    sample_positions_xy_m: Float64Array,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
) -> Float64Array:
    """Ratio of the two source-receiver ranges at every sample position.

    Args:
        sample_positions_xy_m: Float64 array of shape (n, 2).
        receiver_1_xy_m: Float64 array of shape (2,).
        receiver_2_xy_m: Float64 array of shape (2,).

    Returns:
        Float64 array of shape (n,), the ratio `r1 / r2`.
    """
    range_1_m = np.linalg.norm(sample_positions_xy_m - receiver_1_xy_m, axis=-1)
    range_2_m = np.linalg.norm(sample_positions_xy_m - receiver_2_xy_m, axis=-1)
    return np.asarray(
        range_1_m / np.maximum(range_2_m, RANGE_RATIO_FLOOR_M), dtype=np.float64
    )


def compute_near_singular_band_mask(
    sample_positions_xy_m: Float64Array,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    near_singular_tolerance: float,
) -> BoolArray:
    """Mark the positions whose range ratio is too close to unity to invert.

    On the perpendicular bisector both observables vanish at once, so every
    point along it produces the same measurement. Sensitivity falls off as the
    ratio approaches one, which is why the flag covers a band and not a line.

    Args:
        sample_positions_xy_m: Float64 array of shape (n, 2).
        receiver_1_xy_m: Float64 array of shape (2,).
        receiver_2_xy_m: Float64 array of shape (2,).
        near_singular_tolerance: Smallest trusted distance of the ratio from one.

    Returns:
        Bool array of shape (n,), True inside the degenerate band.
    """
    range_ratio = compute_range_ratio_field(
        sample_positions_xy_m, receiver_1_xy_m, receiver_2_xy_m
    )
    return np.asarray(np.abs(range_ratio - 1.0) < near_singular_tolerance)


def compute_perpendicular_bisector_endpoints_xy_m(
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    extent_x_m: float,
    extent_y_m: float,
) -> Float64Array:
    """Endpoints of the receiver pair's perpendicular bisector across the domain.

    Args:
        receiver_1_xy_m: Float64 array of shape (2,).
        receiver_2_xy_m: Float64 array of shape (2,).
        extent_x_m: Domain extent along x, in metres.
        extent_y_m: Domain extent along y, in metres.

    Returns:
        Float64 array of shape (2, 2), the two endpoints.

    Raises:
        ValueError: If the two receivers sit at the same position, where no
            bisector exists.
    """
    baseline_xy_m = receiver_2_xy_m - receiver_1_xy_m
    baseline_m = float(np.linalg.norm(baseline_xy_m))
    if baseline_m <= 0.0:
        raise ValueError("the two receivers must sit at different positions")

    midpoint_xy_m = 0.5 * (receiver_1_xy_m + receiver_2_xy_m)
    normal_xy_m = (
        np.array([-baseline_xy_m[1], baseline_xy_m[0]], dtype=np.float64) / baseline_m
    )
    half_length_m = float(np.hypot(extent_x_m, extent_y_m))
    return np.stack(
        (
            midpoint_xy_m - half_length_m * normal_xy_m,
            midpoint_xy_m + half_length_m * normal_xy_m,
        )
    )


def plot_scene_geometry(
    extent_x_m: float,
    extent_y_m: float,
    ignition_positions_xy_m: Float64Array,
    receiver_positions_xy_m: Float64Array,
    near_singular_tolerance: float,
    title: str,
) -> Figure:
    """Draw one rung of the ladder: domain, ignition points, receivers, bisector.

    The shaded band is where the level ratio is within `near_singular_tolerance`
    of unity for the first receiver pair. A source inside it is unrecoverable
    however clean the signal, which is what turns this from a map into an
    identifiability diagram.

    Args:
        extent_x_m: Domain extent along x, in metres.
        extent_y_m: Domain extent along y, in metres.
        ignition_positions_xy_m: Float64 array of shape (k, 2).
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        near_singular_tolerance: Smallest trusted distance of the ratio from one.
        title: Figure title, naming the scene.

    Returns:
        A matplotlib Figure. The caller saves or displays it.
    """
    figure = Figure(figsize=(3.5, 3.2))
    axes = figure.add_subplot(111)

    if receiver_positions_xy_m.shape[0] >= 2:
        grid_x_m, grid_y_m = np.meshgrid(
            np.linspace(0.0, extent_x_m, BAND_GRID_RESOLUTION),
            np.linspace(0.0, extent_y_m, BAND_GRID_RESOLUTION),
        )
        sample_positions_xy_m = np.stack((grid_x_m.ravel(), grid_y_m.ravel()), axis=1)
        is_degenerate = compute_near_singular_band_mask(
            sample_positions_xy_m,
            receiver_positions_xy_m[0],
            receiver_positions_xy_m[1],
            near_singular_tolerance,
        ).reshape(grid_x_m.shape)
        axes.contourf(
            grid_x_m,
            grid_y_m,
            is_degenerate.astype(np.float64),
            levels=[0.5, 1.5],
            colors=["#d94a3d"],
            alpha=0.18,
        )
        bisector_xy_m = compute_perpendicular_bisector_endpoints_xy_m(
            receiver_positions_xy_m[0],
            receiver_positions_xy_m[1],
            extent_x_m,
            extent_y_m,
        )
        axes.plot(
            bisector_xy_m[:, 0],
            bisector_xy_m[:, 1],
            linestyle="--",
            linewidth=1.0,
            color="#d94a3d",
            label="bisector",
        )

    axes.scatter(
        receiver_positions_xy_m[:, 0],
        receiver_positions_xy_m[:, 1],
        marker="v",
        s=36,
        color="#2f5d8a",
        label="receivers",
        zorder=3,
    )
    axes.scatter(
        ignition_positions_xy_m[:, 0],
        ignition_positions_xy_m[:, 1],
        marker="*",
        s=90,
        color="#e08a1e",
        label="ignition",
        zorder=4,
    )

    axes.set_xlim(0.0, extent_x_m)
    axes.set_ylim(0.0, extent_y_m)
    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title(title)
    axes.legend(loc="upper right", fontsize="x-small", framealpha=0.9)
    figure.tight_layout()
    return figure


def plot_scene_geometry_panels(
    scene_titles: list[str],
    extents_xy_m: list[tuple[float, float]],
    ignition_positions_xy_m: list[Float64Array],
    receiver_positions_xy_m: list[Float64Array],
    near_singular_tolerance: float,
) -> Figure:
    """Draw every rung of the ladder side by side, one panel each.

    Args:
        scene_titles: One title per scene.
        extents_xy_m: One `(extent_x_m, extent_y_m)` pair per scene.
        ignition_positions_xy_m: One `(k, 2)` array per scene.
        receiver_positions_xy_m: One `(n_receivers, 2)` array per scene.
        near_singular_tolerance: Smallest trusted distance of the ratio from one.

    Returns:
        A matplotlib Figure with one axes per scene.

    Raises:
        ValueError: If the four lists do not have the same length.
    """
    lengths = {
        len(scene_titles),
        len(extents_xy_m),
        len(ignition_positions_xy_m),
        len(receiver_positions_xy_m),
    }
    if len(lengths) != 1:
        raise ValueError("every scene must supply a title, an extent and both layouts")

    scene_count = len(scene_titles)
    figure = Figure(figsize=(3.0 * scene_count, 3.2))
    for scene_index in range(scene_count):
        axes = figure.add_subplot(1, scene_count, scene_index + 1)
        extent_x_m, extent_y_m = extents_xy_m[scene_index]
        receivers_xy_m = receiver_positions_xy_m[scene_index]
        if receivers_xy_m.shape[0] >= 2:
            bisector_xy_m = compute_perpendicular_bisector_endpoints_xy_m(
                receivers_xy_m[0], receivers_xy_m[1], extent_x_m, extent_y_m
            )
            axes.plot(
                bisector_xy_m[:, 0],
                bisector_xy_m[:, 1],
                linestyle="--",
                linewidth=1.0,
                color="#d94a3d",
            )
        axes.scatter(
            receivers_xy_m[:, 0],
            receivers_xy_m[:, 1],
            marker="v",
            s=28,
            color="#2f5d8a",
        )
        ignitions_xy_m = ignition_positions_xy_m[scene_index]
        axes.scatter(
            ignitions_xy_m[:, 0],
            ignitions_xy_m[:, 1],
            marker="*",
            s=70,
            color="#e08a1e",
        )
        axes.set_xlim(0.0, extent_x_m)
        axes.set_ylim(0.0, extent_y_m)
        axes.set_aspect("equal")
        axes.set_xlabel("x (m)")
        if scene_index == 0:
            axes.set_ylabel("y (m)")
        axes.set_title(scene_titles[scene_index], fontsize="small")
    figure.tight_layout()
    return figure
