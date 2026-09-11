"""Bearing and extent against time, each beside the truth it is claiming.

Two panels. The bearing panel carries the per-frame estimate, the fused answer
and the truth, so a method that is right on average but wrong every frame is
visible as such. The extent panel carries the array response alongside the
estimate and the truth, because an extent below the response is not a
measurement and the figure should say where that line falls rather than leaving
the reader to infer it.
"""

import numpy as np
from matplotlib.figure import Figure

from src.utils.array_types import BoolArray, Float64Array

TRUTH_COLOR: str = "#e08a1e"
ESTIMATE_COLOR: str = "#2f5d8a"
FUSED_COLOR: str = "#43d6c4"
RESPONSE_COLOR: str = "#999999"
UNRESOLVED_COLOR: str = "#d94a3d"
FIGURE_SIZE_INCHES: tuple[float, float] = (11.0, 4.2)


def plot_fire_characterization(
    times_s: Float64Array,
    estimated_bearings_deg: Float64Array,
    true_bearings_deg: Float64Array,
    fused_bearing_deg: float,
    estimated_semi_axes_m: Float64Array,
    true_semi_axes_m: Float64Array,
    is_resolved: BoolArray,
    response_semi_axes_m: Float64Array,
) -> Figure:
    """Draws the bearing series and the extent series against their truth.

    Args:
        times_s: Frame times, shape `(n_frames,)`.
        estimated_bearings_deg: Per-frame bearing estimate, same shape.
        true_bearings_deg: Per-frame truth, same shape.
        fused_bearing_deg: The run's single fused bearing.
        estimated_semi_axes_m: Per-frame major semi-axis estimate, same shape.
        true_semi_axes_m: Per-frame truth, same shape.
        is_resolved: Whether each frame's extent is a measurement, same shape.
        response_semi_axes_m: The array response's own semi-axis, same shape.

    Returns:
        A matplotlib Figure. Callers save it; this function never does.

    Raises:
        ValueError: If the series are not all the same length.
    """
    lengths = {
        np.asarray(series).size
        for series in (
            times_s,
            estimated_bearings_deg,
            true_bearings_deg,
            estimated_semi_axes_m,
            true_semi_axes_m,
            is_resolved,
            response_semi_axes_m,
        )
    }
    if len(lengths) != 1:
        raise ValueError("every characterisation series must have the same length")

    times = np.asarray(times_s, dtype=np.float64)
    figure = Figure(figsize=FIGURE_SIZE_INCHES)

    bearing_axes = figure.add_subplot(121)
    bearing_axes.plot(
        times,
        np.asarray(true_bearings_deg),
        color=TRUTH_COLOR,
        marker="o",
        markersize=4,
        label="true",
    )
    bearing_axes.plot(
        times,
        np.asarray(estimated_bearings_deg),
        color=ESTIMATE_COLOR,
        marker="x",
        markersize=5,
        linestyle="--",
        label="skewness, per frame",
    )
    bearing_axes.axhline(
        fused_bearing_deg,
        color=FUSED_COLOR,
        linewidth=1.4,
        label=f"fused, {fused_bearing_deg:.1f} deg",
    )
    bearing_axes.set_xlabel("simulated time (s)")
    bearing_axes.set_ylabel("bearing (deg)")
    bearing_axes.set_title("direction of spread")
    bearing_axes.legend(fontsize=7, framealpha=0.8)

    extent_axes = figure.add_subplot(122)
    extent_axes.plot(
        times,
        np.asarray(true_semi_axes_m),
        color=TRUTH_COLOR,
        marker="o",
        markersize=4,
        label="true",
    )
    extent_axes.plot(
        times,
        np.asarray(response_semi_axes_m),
        color=RESPONSE_COLOR,
        linestyle=":",
        label="array response",
    )
    resolved = np.asarray(is_resolved, dtype=bool)
    estimated = np.asarray(estimated_semi_axes_m, dtype=np.float64)
    extent_axes.plot(
        times,
        estimated,
        color=ESTIMATE_COLOR,
        linestyle="--",
        label="estimated",
    )
    if np.any(resolved):
        extent_axes.scatter(
            times[resolved],
            estimated[resolved],
            marker="x",
            s=40,
            color=ESTIMATE_COLOR,
            label="resolved",
        )
    if np.any(~resolved):
        extent_axes.scatter(
            times[~resolved],
            estimated[~resolved],
            marker="v",
            s=32,
            color=UNRESOLVED_COLOR,
            label="upper bound only",
        )
    extent_axes.set_xlabel("simulated time (s)")
    extent_axes.set_ylabel("major semi-axis (m)")
    extent_axes.set_title("front extent")
    extent_axes.legend(fontsize=7, framealpha=0.8)

    figure.suptitle("fire characterisation from the imaging map")
    figure.tight_layout()
    return figure
