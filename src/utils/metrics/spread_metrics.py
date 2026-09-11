"""Scoring a rate of spread and an extent against what the fire actually did.

The true rate is fitted the same way the estimate is — a Theil-Sen slope of the
head position over the same frames — so that only the input differs and the
comparison is not partly a comparison of two fitting methods.
"""

from typing import Any

import numpy as np
from scipy.stats import theilslopes

from src.utils.array_types import Float64Array


def compute_true_rate_of_spread_m_per_s(
    times_s: Float64Array, head_distances_m: Float64Array
) -> float:
    """Fits the true head rate the same way the estimate is fitted.

    Args:
        times_s: Observation times, shape `(n_frames,)`.
        head_distances_m: True head distance from ignition at each, same shape.

    Returns:
        The true head rate in metres per second.

    Raises:
        ValueError: If fewer than two observations are given.
    """
    times = np.asarray(times_s, dtype=np.float64)
    distances = np.asarray(head_distances_m, dtype=np.float64)
    if times.size < 2:
        raise ValueError("a rate needs at least two observations")
    return float(theilslopes(distances, times)[0])


def compute_rate_of_spread_error(
    estimated_m_per_s: float, true_m_per_s: float
) -> dict[str, Any]:
    """Absolute and relative error of one rate.

    Args:
        estimated_m_per_s: What was estimated.
        true_m_per_s: What the fire did.

    Returns:
        The two errors, with the relative one None when the truth is zero.
    """
    absolute = float(estimated_m_per_s) - float(true_m_per_s)
    return {
        "estimated_m_per_s": float(estimated_m_per_s),
        "true_m_per_s": float(true_m_per_s),
        "absolute_error_m_per_s": absolute,
        "relative_error": (
            abs(absolute) / abs(true_m_per_s) if abs(true_m_per_s) > 0.0 else None
        ),
    }


def compute_extent_error_series(
    estimated_semi_axes_m: Float64Array, true_semi_axes_m: Float64Array
) -> Float64Array:
    """Relative error of the major semi-axis at every resolved frame.

    Args:
        estimated_semi_axes_m: Estimates, shape `(n_frames,)`.
        true_semi_axes_m: Truth at the same frames, same shape.

    Returns:
        Relative errors, shape `(n_frames,)`.

    Raises:
        ValueError: If the two series are different lengths.
    """
    estimated = np.asarray(estimated_semi_axes_m, dtype=np.float64)
    true_values = np.asarray(true_semi_axes_m, dtype=np.float64)
    if estimated.size != true_values.size:
        raise ValueError("each estimated semi-axis needs exactly one true semi-axis")
    denominator = np.where(np.abs(true_values) > 0.0, np.abs(true_values), np.nan)
    return np.asarray(np.abs(estimated - true_values) / denominator, dtype=np.float64)


def is_flag_honest(
    is_resolved: bool, true_semi_axis_m: float, response_semi_axis_m: float
) -> bool:
    """Whether an unresolved flag was raised for the stated reason.

    A flag that fires whenever the estimator is unsure is merely conservative.
    The claim being checked is stronger: a frame reported unresolved really did
    have a front smaller than the array's response.

    Args:
        is_resolved: What the extent stage reported.
        true_semi_axis_m: The front's true major semi-axis.
        response_semi_axis_m: The array response's own major semi-axis.

    Returns:
        True when the flag is consistent with the truth.
    """
    if is_resolved:
        return True
    return float(true_semi_axis_m) <= float(response_semi_axis_m)
