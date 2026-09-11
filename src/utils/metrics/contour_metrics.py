"""Scoring an estimated front against the one the fire actually had.

Three metrics because they fail differently. Hausdorff catches the worst local
excursion, the mean radial error reports typical accuracy, and the overlap says
whether the two agree as regions rather than as curves. Any one of them alone
is misleading: a contour that is right everywhere but one spur scores well on
two of the three and badly on Hausdorff, which is the point of reporting all
three.

**The overlap is computed radially, and the plan's enclosed-area form does not
apply here.** It asks for an intersection over union of the areas the two
contours enclose. A radiating fire front that has burnt out behind itself is an
open arc — ninety degrees of three hundred and sixty on `configs/f1` — so
neither contour encloses anything and the ratio is undefined. Taking the
overlap of the sectors swept from the ignition point instead is defined for a
partial arc, reduces to the enclosed-area ratio when the front does close, and
measures the thing a reader wants from it: how much of the burnt ground the two
agree on.
"""

import numpy as np
from scipy.spatial.distance import cdist

from src.utils.array_types import Float64Array


def compute_hausdorff_distance(
    estimated_contour_xy_m: Float64Array, true_contour_xy_m: Float64Array
) -> float:
    """Largest distance from either contour to the nearest point of the other.

    Symmetric, so a contour that covers only part of the truth is penalised for
    what it leaves out as well as for where it strays.

    Args:
        estimated_contour_xy_m: Estimated points, shape `(n_estimated, 2)`.
        true_contour_xy_m: True points, shape `(n_true, 2)`.

    Returns:
        The distance in metres, infinite when either contour is empty.
    """
    estimated = np.atleast_2d(np.asarray(estimated_contour_xy_m, dtype=np.float64))
    true_points = np.atleast_2d(np.asarray(true_contour_xy_m, dtype=np.float64))
    if estimated.size == 0 or true_points.size == 0:
        return float("inf")
    distances = cdist(estimated, true_points)
    return float(
        max(
            float(np.max(np.min(distances, axis=1))),
            float(np.max(np.min(distances, axis=0))),
        )
    )


def compute_mean_radial_error(
    estimated_distances_m: Float64Array, true_distances_m: Float64Array
) -> float:
    """Mean absolute difference in radius, over directions both cover.

    Directions only one of the two covers are excluded rather than counted as
    zero error, and the coverage agreement is reported separately so that
    exclusion cannot flatter a contour that covers almost nothing.

    Args:
        estimated_distances_m: Estimated radius per angle, `nan` where absent.
        true_distances_m: True radius per angle, `nan` where absent.

    Returns:
        The mean absolute error in metres, `nan` when they share no direction.

    Raises:
        ValueError: If the two series are different lengths.
    """
    estimated = np.asarray(estimated_distances_m, dtype=np.float64)
    true_values = np.asarray(true_distances_m, dtype=np.float64)
    if estimated.size != true_values.size:
        raise ValueError("each estimated distance needs exactly one true distance")
    shared = np.isfinite(estimated) & np.isfinite(true_values)
    if not bool(np.any(shared)):
        return float("nan")
    return float(np.mean(np.abs(estimated[shared] - true_values[shared])))


def compute_sector_overlap(
    estimated_distances_m: Float64Array, true_distances_m: Float64Array
) -> float:
    """Intersection over union of the swept sectors, from the ignition point.

    Each direction contributes a wedge whose area goes as the square of its
    radius, so the ratio is the area both agree on over the area either claims.
    A direction one covers and the other does not contributes to the union
    alone, which is what makes missing coverage cost something.

    Args:
        estimated_distances_m: Estimated radius per angle, `nan` where absent.
        true_distances_m: True radius per angle, `nan` where absent.

    Returns:
        The ratio in `[0, 1]`, zero when neither claims any ground.

    Raises:
        ValueError: If the two series are different lengths.
    """
    estimated = np.asarray(estimated_distances_m, dtype=np.float64)
    true_values = np.asarray(true_distances_m, dtype=np.float64)
    if estimated.size != true_values.size:
        raise ValueError("each estimated distance needs exactly one true distance")
    estimated_area = np.where(np.isfinite(estimated), estimated, 0.0) ** 2
    true_area = np.where(np.isfinite(true_values), true_values, 0.0) ** 2
    union = float(np.sum(np.maximum(estimated_area, true_area)))
    if union <= 0.0:
        return 0.0
    return float(np.sum(np.minimum(estimated_area, true_area)) / union)


def compute_coverage_agreement(
    estimated_distances_m: Float64Array, true_distances_m: Float64Array
) -> float:
    """Share of directions on which the two agree that a front is present.

    Reported alongside the radial error, which is taken only over shared
    directions and would otherwise flatter a contour covering almost nothing.

    Args:
        estimated_distances_m: Estimated radius per angle, `nan` where absent.
        true_distances_m: True radius per angle, `nan` where absent.

    Returns:
        Fraction of directions where presence matches.

    Raises:
        ValueError: If the two series are different lengths.
    """
    estimated = np.asarray(estimated_distances_m, dtype=np.float64)
    true_values = np.asarray(true_distances_m, dtype=np.float64)
    if estimated.size != true_values.size:
        raise ValueError("each estimated distance needs exactly one true distance")
    if estimated.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(estimated) == np.isfinite(true_values)))
