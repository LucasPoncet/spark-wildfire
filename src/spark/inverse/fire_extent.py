"""How large the fire is, from a map that need not resolve it.

The relation that makes this possible is that a thin ring of radius `R` has
second central moment `R^2 / 2` about each axis. The second moment knows the
ring's radius without the map ever showing a hole, so an extent survives a
front several times narrower than the array can resolve.

What has to be removed first is the array's own contribution:

    Sigma_source = Sigma_observed - Sigma_response

Both sides are measured through the identical preparation, which is the only
reason the difference means anything.

Where that difference goes non-positive the front is smaller than the response
along that axis. The honest output there is a flag and an upper bound, never a
clipped number presented as a measurement: a negative eigenvalue silently set
to zero turns "too small to measure" into a fabricated extent, and on an early
frame it will happen every time.
"""

from dataclasses import dataclass

import numpy as np

from src.spark.inverse.map_moments import compute_map_moments
from src.utils.array_types import Float64Array

THIN_RING_SHAPE_FACTOR: float = float(np.sqrt(2.0))


@dataclass(frozen=True)
class ExtentEstimate:
    """One frame's extent, and whether it is a measurement or a ceiling.

    Attributes:
        time_s: Frame time.
        centroid_xy_m: Map centroid, shape `(2,)`.
        semi_axis_major_m: Major semi-axis, in metres.
        semi_axis_minor_m: Minor semi-axis, in metres.
        orientation_rad: Direction of the major axis, modulo half a turn.
        is_resolved: Whether the front exceeded the response on both axes.
        upper_bound_only: Whether the semi-axes are ceilings rather than
            measurements, which is the case whenever it is not resolved.
        observed_semi_axis_major_m: Major semi-axis before the response is
            subtracted, kept so a ceiling can be read even when resolved.
    """

    time_s: float
    centroid_xy_m: Float64Array
    semi_axis_major_m: float
    semi_axis_minor_m: float
    orientation_rad: float
    is_resolved: bool
    upper_bound_only: bool
    observed_semi_axis_major_m: float


def estimate_fire_extent(
    time_s: float,
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    background_percentile: float,
    point_spread_covariance_m2: Float64Array,
    shape_factor: float,
) -> ExtentEstimate:
    """Deconvolves the array response out of one frame's second moment.

    Args:
        time_s: Frame time.
        map_values: The frame, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        background_percentile: Percentile of the frame treated as background.
        point_spread_covariance_m2: The response's covariance at this centroid.
        shape_factor: Multiplier from the root of an eigenvalue to a semi-axis.
            The root of two is the thin circular ring.

    Returns:
        The frame's extent, flagged when it is a ceiling rather than a
        measurement.
    """
    moments = compute_map_moments(
        map_values, candidate_positions_xyz_m, background_percentile
    )
    observed_eigenvalues = np.linalg.eigvalsh(moments.covariance_xy_m2)
    observed_major_m = shape_factor * float(
        np.sqrt(np.clip(np.max(observed_eigenvalues), 0.0, None))
    )

    source_covariance_m2 = moments.covariance_xy_m2 - np.asarray(
        point_spread_covariance_m2, dtype=np.float64
    )
    eigenvalues, eigenvectors = np.linalg.eigh(source_covariance_m2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major_direction = eigenvectors[:, order[0]]
    is_resolved = bool(np.min(eigenvalues) > 0.0)

    if not is_resolved:
        return ExtentEstimate(
            time_s=time_s,
            centroid_xy_m=moments.centroid_xy_m,
            semi_axis_major_m=observed_major_m,
            semi_axis_minor_m=shape_factor
            * float(np.sqrt(np.clip(np.min(observed_eigenvalues), 0.0, None))),
            orientation_rad=float(
                np.arctan2(moments.principal_axes[1, 0], moments.principal_axes[0, 0])
            ),
            is_resolved=False,
            upper_bound_only=True,
            observed_semi_axis_major_m=observed_major_m,
        )
    return ExtentEstimate(
        time_s=time_s,
        centroid_xy_m=moments.centroid_xy_m,
        semi_axis_major_m=shape_factor * float(np.sqrt(eigenvalues[0])),
        semi_axis_minor_m=shape_factor * float(np.sqrt(eigenvalues[1])),
        orientation_rad=float(np.arctan2(major_direction[1], major_direction[0])),
        is_resolved=True,
        upper_bound_only=False,
        observed_semi_axis_major_m=observed_major_m,
    )


def select_resolved(extents: list[ExtentEstimate]) -> list[ExtentEstimate]:
    """Keeps only the frames whose extent is a measurement.

    Args:
        extents: Every frame's extent.

    Returns:
        Those flagged resolved, in the order given.
    """
    return [extent for extent in extents if extent.is_resolved]
