"""Where the front is, in every direction and over time.

Two panels. The left is polar in all but projection: front distance against
bearing, with the truth and both estimates on one pair of axes, so a route that
is right in the head direction and wrong on the flanks reads as such rather
than averaging to a single number. The right follows the head distance over the
run with the array response drawn across it, because a head distance below the
response is not a measurement and the figure should place that line.

Directions the front does not cover carry no distance rather than a zero, so
the lines simply stop. The gap is information — it is the angular extent that
radiates — and filling it would hide the one thing this scene's geometry makes
unusual.
"""

import numpy as np
from matplotlib.figure import Figure

from src.utils.array_types import Float64Array

TRUTH_COLOR: str = "#e08a1e"
PERIMETER_COLOR: str = "#2f5d8a"
DECONVOLUTION_COLOR: str = "#8e5fb0"
RESPONSE_COLOR: str = "#999999"
FIGURE_SIZE_INCHES: tuple[float, float] = (11.5, 4.3)


def plot_front_position(
    angles_rad: Float64Array,
    true_distances_m: Float64Array,
    perimeter_distances_m: Float64Array,
    deconvolved_distances_m: Float64Array,
    times_s: Float64Array,
    profile_head_distances_m: Float64Array,
    true_head_distances_m: Float64Array,
    response_semi_axes_m: Float64Array,
) -> Figure:
    """Draws distance against bearing, and head distance against time.

    Args:
        angles_rad: Directions the front was evaluated on.
        true_distances_m: True radius per angle, `nan` where uncovered.
        perimeter_distances_m: Parametric arc's radius per angle.
        deconvolved_distances_m: Deconvolution's radius per angle.
        times_s: Frame times, shape `(n_frames,)`.
        profile_head_distances_m: Matched-filter head distance per frame.
        true_head_distances_m: True head distance per frame.
        response_semi_axes_m: The array response's semi-axis per frame.

    Returns:
        A matplotlib Figure. Callers save it; this function never does.

    Raises:
        ValueError: If the angular series are not all the same length.
    """
    lengths = {
        np.asarray(series).size
        for series in (
            angles_rad,
            true_distances_m,
            perimeter_distances_m,
            deconvolved_distances_m,
        )
    }
    if len(lengths) != 1:
        raise ValueError("every angular series must have the same length")

    degrees = np.rad2deg(np.asarray(angles_rad, dtype=np.float64))
    figure = Figure(figsize=FIGURE_SIZE_INCHES)

    angular_axes = figure.add_subplot(121)
    angular_axes.plot(
        degrees,
        np.asarray(true_distances_m),
        color=TRUTH_COLOR,
        linewidth=2.0,
        label="true front",
    )
    angular_axes.plot(
        degrees,
        np.asarray(perimeter_distances_m),
        color=PERIMETER_COLOR,
        linestyle="--",
        label="parametric arc",
    )
    angular_axes.plot(
        degrees,
        np.asarray(deconvolved_distances_m),
        color=DECONVOLUTION_COLOR,
        linestyle=":",
        linewidth=1.8,
        label="deconvolution",
    )
    angular_axes.set_xlabel("bearing (deg)")
    angular_axes.set_ylabel("front distance from ignition (m)")
    angular_axes.set_title("front distance by direction, final frame")
    angular_axes.legend(fontsize=7, framealpha=0.8)

    time_axes = figure.add_subplot(122)
    times = np.asarray(times_s, dtype=np.float64)
    time_axes.plot(
        times,
        np.asarray(true_head_distances_m),
        color=TRUTH_COLOR,
        marker="o",
        markersize=4,
        label="true head",
    )
    time_axes.plot(
        times,
        np.asarray(profile_head_distances_m),
        color=PERIMETER_COLOR,
        marker="x",
        markersize=5,
        linestyle="--",
        label="matched filter",
    )
    time_axes.plot(
        times,
        np.asarray(response_semi_axes_m),
        color=RESPONSE_COLOR,
        linestyle=":",
        label="array response",
    )
    time_axes.set_xlabel("simulated time (s)")
    time_axes.set_ylabel("head distance (m)")
    time_axes.set_title("head distance over the run")
    time_axes.legend(fontsize=7, framealpha=0.8)

    figure.suptitle("front position from the imaging map")
    figure.tight_layout()
    return figure
