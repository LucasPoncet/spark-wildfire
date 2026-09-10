"""The steered response power map, with what the search made of it."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from src.utils.array_types import Float64Array

MAP_COLORMAP: str = "magma"
ELLIPSE_SIGMA_COUNT: float = 1.0


def reshape_map_to_grid(
    steered_response_power_map: Float64Array,
    candidate_positions_xyz_m: Float64Array,
) -> tuple[Float64Array, Float64Array, Float64Array]:
    """Folds a flat map back into the rectangular grid it came from.

    Args:
        steered_response_power_map: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.

    Returns:
        `(x_m, y_m, map_grid)` ready for `pcolormesh`.

    Raises:
        ValueError: If the positions do not form a rectangular grid.
    """
    positions = np.atleast_2d(np.asarray(candidate_positions_xyz_m, dtype=np.float64))
    x_m = np.unique(positions[:, 0])
    y_m = np.unique(positions[:, 1])
    if x_m.size * y_m.size != positions.shape[0]:
        raise ValueError("the candidate positions do not form a rectangular grid")
    return (
        x_m,
        y_m,
        np.asarray(steered_response_power_map, dtype=np.float64).reshape(
            y_m.size, x_m.size
        ),
    )


def add_error_ellipse(
    axes: Axes,
    position_xy_m: Float64Array,
    position_covariance_m2: Float64Array,
    edge_color: str,
) -> None:
    """Draws the one-sigma error ellipse of one estimate.

    Args:
        axes: Axes to draw on.
        position_xy_m: Centre of the ellipse, in metres.
        position_covariance_m2: Two-by-two covariance in metres squared.
        edge_color: Colour of the ellipse edge.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.asarray(position_covariance_m2, dtype=np.float64)
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    major_direction = eigenvectors[:, order[0]]
    axes.add_patch(
        Ellipse(
            xy=(float(position_xy_m[0]), float(position_xy_m[1])),
            width=2.0 * ELLIPSE_SIGMA_COUNT * float(np.sqrt(eigenvalues[0])),
            height=2.0 * ELLIPSE_SIGMA_COUNT * float(np.sqrt(eigenvalues[1])),
            angle=float(np.degrees(np.arctan2(major_direction[1], major_direction[0]))),
            facecolor="none",
            edgecolor=edge_color,
            linewidth=1.2,
        )
    )


def plot_steered_response_power_map(
    steered_response_power_map: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    true_source_positions_xy_m: Float64Array,
    estimated_positions_xy_m: Float64Array,
    estimated_covariances_m2: list[Float64Array],
    title: str,
) -> Figure:
    """Draws one search map with the receivers, the truth and the estimates.

    Args:
        steered_response_power_map: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        true_source_positions_xy_m: True positions, shape `(n_true, 2)`.
        estimated_positions_xy_m: Estimated positions, shape `(n_estimated, 2)`.
        estimated_covariances_m2: One covariance per estimate.
        title: Figure title.

    Returns:
        The figure. Saving it is the caller's job.
    """
    x_m, y_m, map_grid = reshape_map_to_grid(
        steered_response_power_map, candidate_positions_xyz_m
    )
    figure = Figure(figsize=(7.0, 6.0))
    axes = figure.add_subplot(111)
    mesh = axes.pcolormesh(x_m, y_m, map_grid, cmap=MAP_COLORMAP, shading="nearest")
    figure.colorbar(mesh, ax=axes, label="steered response power")

    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    axes.scatter(
        receivers[:, 0],
        receivers[:, 1],
        marker="v",
        s=70,
        color="deepskyblue",
        edgecolor="black",
        label="receivers",
        zorder=3,
    )
    true_positions = np.atleast_2d(
        np.asarray(true_source_positions_xy_m, dtype=np.float64)
    )
    if true_positions.size:
        axes.scatter(
            true_positions[:, 0],
            true_positions[:, 1],
            marker="*",
            s=180,
            color="white",
            edgecolor="black",
            label="true sources",
            zorder=4,
        )
    estimates = np.asarray(estimated_positions_xy_m, dtype=np.float64)
    if estimates.size:
        estimates = np.atleast_2d(estimates)
        axes.scatter(
            estimates[:, 0],
            estimates[:, 1],
            marker="o",
            s=60,
            facecolor="none",
            edgecolor="lime",
            linewidth=1.6,
            label="estimates",
            zorder=5,
        )
        for position_xy_m, covariance_m2 in zip(
            estimates, estimated_covariances_m2, strict=True
        ):
            add_error_ellipse(axes, position_xy_m, covariance_m2, "lime")

    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title(title)
    axes.set_aspect("equal")
    axes.legend(loc="upper right", framealpha=0.8)
    figure.tight_layout()
    return figure


def plot_windowed_position_clusters(
    window_maxima_xy_m: Float64Array,
    cluster_centroids_xy_m: Float64Array,
    true_source_positions_xy_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
) -> Figure:
    """Draws the per-window maxima and the clusters they formed.

    Args:
        window_maxima_xy_m: Per-window maxima, shape `(n_windows, 2)`.
        cluster_centroids_xy_m: Cluster centroids, shape `(n_clusters, 2)`.
        true_source_positions_xy_m: True positions, shape `(n_true, 2)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.

    Returns:
        The figure. Saving it is the caller's job.
    """
    maxima = np.atleast_2d(np.asarray(window_maxima_xy_m, dtype=np.float64))
    figure = Figure(figsize=(7.0, 6.0))
    axes = figure.add_subplot(111)
    axes.scatter(
        maxima[:, 0],
        maxima[:, 1],
        s=12,
        alpha=0.35,
        color="tab:blue",
        label="per-window maxima",
    )
    centroids = np.asarray(cluster_centroids_xy_m, dtype=np.float64)
    if centroids.size:
        centroids = np.atleast_2d(centroids)
        axes.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="o",
            s=90,
            facecolor="none",
            edgecolor="tab:red",
            linewidth=1.8,
            label="cluster centroids",
        )
    true_positions = np.atleast_2d(
        np.asarray(true_source_positions_xy_m, dtype=np.float64)
    )
    if true_positions.size:
        axes.scatter(
            true_positions[:, 0],
            true_positions[:, 1],
            marker="*",
            s=180,
            color="black",
            label="true sources",
        )
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    axes.scatter(
        receivers[:, 0],
        receivers[:, 1],
        marker="v",
        s=70,
        color="tab:green",
        label="receivers",
    )
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title("windowed single-source maxima and their clusters")
    axes.set_aspect("equal")
    axes.legend(loc="upper right", framealpha=0.8)
    figure.tight_layout()
    return figure
