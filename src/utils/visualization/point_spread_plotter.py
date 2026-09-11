"""The array response tabulated across the domain, and its validation.

Three panels: where the response displaces a centroid, how wide it is, and how
well the analytic response reproduced a rendered one at the positions that were
checked.

The bias panel is drawn as arrows at their own scale with the scale stated,
because the offsets are sub-metre over a sixty-metre domain and arrows drawn to
scale would be invisible. What the panel is for is the *pattern* — a bias field
that grows outward and turns with the geometry is being sampled correctly, one
that changes direction between neighbours is noise being tabulated.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.utils.array_types import Float64Array

RECEIVER_COLOR: str = "#2f5d8a"
ARROW_COLOR: str = "#d94a3d"
VALIDATION_COLOR: str = "#43a047"
WIDTH_COLORMAP: str = "viridis"
FIGURE_SIZE_INCHES: tuple[float, float] = (13.0, 4.2)


def plot_point_spread_calibration(
    node_positions_xy_m: Float64Array,
    offsets_xy_m: Float64Array,
    semi_axes_m: Float64Array,
    grid_shape: tuple[int, int],
    receiver_positions_xyz_m: Float64Array,
    validation_correlations: Float64Array,
    validation_positions_xy_m: Float64Array,
) -> Figure:
    """Draws the bias field, the width field and the render validation.

    Args:
        node_positions_xy_m: Calibration nodes, shape `(n_nodes, 2)`.
        offsets_xy_m: Centroid offset at each node, shape `(n_nodes, 2)`.
        semi_axes_m: Major response semi-axis at each node, shape `(n_nodes,)`.
        grid_shape: `(n_y, n_x)` of the node grid.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        validation_correlations: Analytic-to-rendered correlation at each
            validation position, shape `(n_validations,)`.
        validation_positions_xy_m: Those positions, shape `(n_validations, 2)`.

    Returns:
        A matplotlib Figure. Callers save it; this function never does.

    Raises:
        ValueError: If the node grid shape does not match the node count.
    """
    nodes = np.atleast_2d(np.asarray(node_positions_xy_m, dtype=np.float64))
    offsets = np.atleast_2d(np.asarray(offsets_xy_m, dtype=np.float64))
    widths_m = np.asarray(semi_axes_m, dtype=np.float64)
    if grid_shape[0] * grid_shape[1] != nodes.shape[0]:
        raise ValueError("the node grid shape does not match the number of nodes")

    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    figure = Figure(figsize=FIGURE_SIZE_INCHES)
    _draw_bias_panel(figure.add_subplot(131), nodes, offsets, receivers)
    _draw_width_panel(figure.add_subplot(132), nodes, widths_m, grid_shape, receivers)
    _draw_validation_panel(
        figure.add_subplot(133),
        np.asarray(validation_correlations, dtype=np.float64),
        np.atleast_2d(np.asarray(validation_positions_xy_m, dtype=np.float64)),
    )
    figure.suptitle("array response calibration")
    figure.tight_layout()
    return figure


def _draw_bias_panel(
    axes: Axes, nodes: Float64Array, offsets: Float64Array, receivers: Float64Array
) -> None:
    """Draws the centroid offset at each node as an arrow.

    Args:
        axes: Axes to draw on.
        nodes: Calibration nodes, shape `(n_nodes, 2)`.
        offsets: Offset at each node, shape `(n_nodes, 2)`.
        receivers: The array, shape `(n_receivers, 3)`.
    """
    magnitudes_m = np.linalg.norm(offsets, axis=1)
    largest_m = float(np.max(magnitudes_m)) if magnitudes_m.size else 0.0
    axes.quiver(
        nodes[:, 0],
        nodes[:, 1],
        offsets[:, 0],
        offsets[:, 1],
        color=ARROW_COLOR,
        angles="xy",
    )
    axes.scatter(
        receivers[:, 0], receivers[:, 1], marker="v", s=22, color=RECEIVER_COLOR
    )
    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title(f"centroid bias, largest {largest_m:.2f} m")


def _draw_width_panel(
    axes: Axes,
    nodes: Float64Array,
    widths_m: Float64Array,
    grid_shape: tuple[int, int],
    receivers: Float64Array,
) -> None:
    """Draws the major response semi-axis as a heat map.

    Args:
        axes: Axes to draw on.
        nodes: Calibration nodes, shape `(n_nodes, 2)`.
        widths_m: Major semi-axis at each node, shape `(n_nodes,)`.
        grid_shape: `(n_y, n_x)` of the node grid.
        receivers: The array, shape `(n_receivers, 3)`.
    """
    node_x_m = np.unique(nodes[:, 0])
    node_y_m = np.unique(nodes[:, 1])
    mesh = axes.pcolormesh(
        node_x_m,
        node_y_m,
        widths_m.reshape(grid_shape),
        cmap=WIDTH_COLORMAP,
        shading="nearest",
    )
    figure = axes.get_figure()
    if figure is not None:
        figure.colorbar(mesh, ax=axes, label="major semi-axis (m)")
    axes.scatter(
        receivers[:, 0], receivers[:, 1], marker="v", s=22, color=RECEIVER_COLOR
    )
    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_title("response width")


def _draw_validation_panel(
    axes: Axes,
    correlations: Float64Array,
    positions_xy_m: Float64Array,
) -> None:
    """Draws the analytic-to-rendered agreement at each checked position.

    Args:
        axes: Axes to draw on.
        correlations: Correlation at each position, shape `(n_positions,)`.
        positions_xy_m: Those positions, shape `(n_positions, 2)`.
    """
    labels = [f"({position[0]:.0f}, {position[1]:.0f})" for position in positions_xy_m]
    axes.bar(labels, correlations, color=VALIDATION_COLOR)
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel("validation position (m)")
    axes.set_ylabel("analytic against rendered")
    axes.set_title("response validation")
