"""Scoring a direction against the direction the fire actually took.

Every difference here is wrapped to half a turn before anything is done with
it. A bearing error computed linearly reports 340 degrees where the estimate
was 20 degrees off, and an average of such errors is meaningless.
"""

from typing import Any

import numpy as np

from src.utils.array_types import Float64Array

WITHIN_TIGHT_DEGREES: float = 10.0
WITHIN_LOOSE_DEGREES: float = 20.0


def compute_circular_error_rad(
    estimated_bearing_rad: float, true_bearing_rad: float
) -> float:
    """Signed difference between two directions, wrapped to half a turn.

    Args:
        estimated_bearing_rad: What was estimated.
        true_bearing_rad: What the fire did.

    Returns:
        The wrapped difference in radians.
    """
    difference = float(estimated_bearing_rad) - float(true_bearing_rad)
    return float(np.arctan2(np.sin(difference), np.cos(difference)))


def compute_bearing_error_series(
    estimated_bearings_rad: Float64Array, true_bearings_rad: Float64Array
) -> Float64Array:
    """Wrapped error at every frame.

    Args:
        estimated_bearings_rad: Estimates, shape `(n_frames,)`.
        true_bearings_rad: Truth at the same frames, same shape.

    Returns:
        Wrapped errors in radians, shape `(n_frames,)`.

    Raises:
        ValueError: If the two series are different lengths.
    """
    estimated = np.asarray(estimated_bearings_rad, dtype=np.float64)
    true_values = np.asarray(true_bearings_rad, dtype=np.float64)
    if estimated.size != true_values.size:
        raise ValueError("each estimated bearing needs exactly one true bearing")
    difference = estimated - true_values
    return np.asarray(
        np.arctan2(np.sin(difference), np.cos(difference)), dtype=np.float64
    )


def summarize_bearing_errors(errors_rad: Float64Array) -> dict[str, Any]:
    """Reduces an error series to what a gate is read against.

    Args:
        errors_rad: Wrapped errors, shape `(n_frames,)`.

    Returns:
        Median and ninetieth-percentile absolute error in degrees, the share of
        frames inside each tolerance, and the frame count.
    """
    absolute_degrees = np.abs(np.rad2deg(np.asarray(errors_rad, dtype=np.float64)))
    if absolute_degrees.size == 0:
        return {
            "frame_count": 0,
            "median_absolute_error_deg": None,
            "ninetieth_percentile_absolute_error_deg": None,
            "fraction_within_10_deg": 0.0,
            "fraction_within_20_deg": 0.0,
        }
    return {
        "frame_count": int(absolute_degrees.size),
        "median_absolute_error_deg": float(np.median(absolute_degrees)),
        "ninetieth_percentile_absolute_error_deg": float(
            np.percentile(absolute_degrees, 90.0)
        ),
        "fraction_within_10_deg": float(
            np.mean(absolute_degrees <= WITHIN_TIGHT_DEGREES)
        ),
        "fraction_within_20_deg": float(
            np.mean(absolute_degrees <= WITHIN_LOOSE_DEGREES)
        ),
    }


def compute_agreement_fraction(
    first_bearings_rad: Float64Array,
    second_bearings_rad: Float64Array,
    tolerance_deg: float,
) -> float:
    """Share of frames on which two estimators agree.

    Two methods that fail differently agreeing is evidence; the same two
    disagreeing is a question about the scene. Either way it is worth a number
    rather than an impression.

    Args:
        first_bearings_rad: One method's estimates, shape `(n_frames,)`.
        second_bearings_rad: Another's, same shape.
        tolerance_deg: How close counts as agreement.

    Returns:
        Fraction of frames within the tolerance.
    """
    errors_rad = compute_bearing_error_series(first_bearings_rad, second_bearings_rad)
    return float(np.mean(np.abs(np.rad2deg(errors_rad)) <= tolerance_deg))
