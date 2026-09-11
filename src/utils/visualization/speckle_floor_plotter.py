"""The grain a single source leaves across the map, against frame duration.

The trade the frame duration sits on has two sides and only one of them is
visible here. A longer frame averages more correlation windows and lowers this
floor; it also smears a moving front over further ground, which no single-frame
measurement can show. The configured duration is marked so the figure says
where the run actually sits on the curve it is trading along.
"""

import numpy as np
from matplotlib.figure import Figure

from src.utils.array_types import Float64Array

CURVE_COLOR: str = "#2f5d8a"
MARKER_COLOR: str = "#d94a3d"
FIGURE_SIZE_INCHES: tuple[float, float] = (5.2, 3.6)


def plot_speckle_floor(
    frame_durations_s: Float64Array,
    speckle_floors: Float64Array,
    configured_frame_duration_s: float,
) -> Figure:
    """Draws the speckle floor against frame duration on log axes.

    Args:
        frame_durations_s: Frame lengths measured at, shape `(n_points,)`.
        speckle_floors: Coefficient of variation at each, same shape.
        configured_frame_duration_s: The duration the scene actually uses,
            marked on the curve.

    Returns:
        A matplotlib Figure. Callers save it; this function never does.

    Raises:
        ValueError: If the two columns are not the same length.
    """
    durations_s = np.asarray(frame_durations_s, dtype=np.float64)
    floors = np.asarray(speckle_floors, dtype=np.float64)
    if durations_s.size != floors.size:
        raise ValueError("each frame duration needs exactly one speckle floor")

    figure = Figure(figsize=FIGURE_SIZE_INCHES)
    axes = figure.add_subplot(111)
    axes.plot(durations_s, floors, marker="o", color=CURVE_COLOR)
    axes.axvline(
        configured_frame_duration_s,
        color=MARKER_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=f"configured, {configured_frame_duration_s:.1f} s",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("frame duration (s)")
    axes.set_ylabel("speckle floor (standard deviation over mean)")
    axes.set_title("map grain against accumulation length")
    axes.legend(fontsize=8, framealpha=0.8)
    figure.tight_layout()
    return figure
