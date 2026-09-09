from dataclasses import dataclass

import numpy as np

from config.simulation_configuration import TriangulationConfiguration
from utils.array_types import Float64Array

LEVEL_RATIO_FLOOR: float = 1e-9


@dataclass(frozen=True)
class TriangulatedPosition:
    """A source position recovered from a level ratio and a path difference.

    Attributes:
        position_xy_m: Estimated position `(x, y)` in metres, in the grid frame.
        position_covariance_m2: Two-by-two position covariance in metres squared.
        range_receiver_1_m: Estimated range to receiver 1, in metres.
        range_receiver_2_m: Estimated range to receiver 2, in metres.
        is_near_singular: Whether the source lies near the perpendicular bisector,
            where both observables vanish and the ranges are not determined.
    """

    position_xy_m: Float64Array
    position_covariance_m2: Float64Array
    range_receiver_1_m: float
    range_receiver_2_m: float
    is_near_singular: bool


def compute_level_ratio(geometric_level_difference_db: float) -> float:
    """Converts a geometric level difference into a range ratio.

    Args:
        geometric_level_difference_db: Level difference in decibels, absorption
            already removed.

    Returns:
        The ratio of range 2 to range 1, dimensionless.
    """
    return float(10.0 ** (geometric_level_difference_db / 20.0))


def is_near_perpendicular_bisector(
    geometric_level_difference_db: float,
    near_singular_tolerance: float,
) -> bool:
    """Reports whether the source is too close to the perpendicular bisector.

    On the bisector both the path difference and the level ratio carry no range
    information at all: every point along it produces the same pair of observables.
    The estimate is degenerate there, not merely imprecise.

    Args:
        geometric_level_difference_db: Level difference in decibels.
        near_singular_tolerance: Smallest trusted distance of the ratio from unity.

    Returns:
        True when the estimate should be treated as low-confidence.
    """
    return (
        abs(compute_level_ratio(geometric_level_difference_db) - 1.0)
        < near_singular_tolerance
    )


def compute_ranges_from_level_ratio(
    geometric_level_difference_db: float,
    path_difference_m: float,
) -> tuple[float, float]:
    """Recovers both ranges from their ratio and their difference.

    Args:
        geometric_level_difference_db: Level difference in decibels, giving the ratio.
        path_difference_m: Range difference in metres, from the delay estimate.

    Returns:
        `(range_receiver_1_m, range_receiver_2_m)` in metres. Near the bisector the
        denominator is floored, so callers must check the near-singular flag.
    """
    level_ratio = compute_level_ratio(geometric_level_difference_db)
    denominator = level_ratio - 1.0
    if abs(denominator) < LEVEL_RATIO_FLOOR:
        denominator = np.copysign(
            LEVEL_RATIO_FLOOR, denominator if denominator != 0.0 else 1.0
        )
    range_receiver_1_m = path_difference_m / denominator
    return float(range_receiver_1_m), float(level_ratio * range_receiver_1_m)


def triangulate_position_xy(
    range_receiver_1_m: float,
    range_receiver_2_m: float,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    domain_side_sign: float,
) -> Float64Array:
    """Places a source from its two ranges, in the frame of the receiver baseline.

    Args:
        range_receiver_1_m: Range to receiver 1, in metres.
        range_receiver_2_m: Range to receiver 2, in metres.
        receiver_1_xy_m: Receiver 1 position `(x, y)` in metres.
        receiver_2_xy_m: Receiver 2 position `(x, y)` in metres.
        domain_side_sign: Which side of the baseline the domain lies on. Two ranges
            admit a mirror pair; placing both receivers on one domain edge puts the
            mirror image outside the grid.

    Returns:
        Position `(x, y)` in metres, in the grid frame.

    Raises:
        ValueError: If the two receivers occupy the same position.
    """
    receiver_1 = np.asarray(receiver_1_xy_m, dtype=np.float64)
    receiver_2 = np.asarray(receiver_2_xy_m, dtype=np.float64)
    baseline_vector = receiver_2 - receiver_1
    baseline_m = float(np.linalg.norm(baseline_vector))
    if baseline_m <= 0.0:
        raise ValueError("the two receivers must occupy distinct positions")

    unit_along_baseline = baseline_vector / baseline_m
    unit_across_baseline = np.array(
        [-unit_along_baseline[1], unit_along_baseline[0]], dtype=np.float64
    )
    local_x_m = (range_receiver_1_m**2 - range_receiver_2_m**2 + baseline_m**2) / (
        2.0 * baseline_m
    )
    local_y_m = np.sqrt(max(range_receiver_1_m**2 - local_x_m**2, 0.0))

    position_xy_m = (
        receiver_1
        + local_x_m * unit_along_baseline
        + domain_side_sign * local_y_m * unit_across_baseline
    )
    return np.asarray(position_xy_m, dtype=np.float64)


def compute_position_from_level_ratio(
    geometric_level_difference_db: float,
    path_difference_m: float,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    domain_side_sign: float,
) -> Float64Array:
    """Goes straight from the two observables to a position.

    Args:
        geometric_level_difference_db: Level difference in decibels.
        path_difference_m: Range difference in metres.
        receiver_1_xy_m: Receiver 1 position `(x, y)` in metres.
        receiver_2_xy_m: Receiver 2 position `(x, y)` in metres.
        domain_side_sign: Which side of the baseline the domain lies on.

    Returns:
        Position `(x, y)` in metres, in the grid frame.
    """
    range_1_m, range_2_m = compute_ranges_from_level_ratio(
        geometric_level_difference_db, path_difference_m
    )
    return triangulate_position_xy(
        range_1_m, range_2_m, receiver_1_xy_m, receiver_2_xy_m, domain_side_sign
    )


def compute_position_covariance(
    geometric_level_difference_db: float,
    path_difference_m: float,
    geometric_level_difference_variance_db2: float,
    path_difference_variance_m2: float,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    configuration: TriangulationConfiguration,
) -> Float64Array:
    """Propagates the observable variances into a position covariance.

    The map from the two observables to a position is differentiated by central
    finite difference, giving a two-by-two Jacobian.

    Args:
        geometric_level_difference_db: Level difference in decibels.
        path_difference_m: Range difference in metres.
        geometric_level_difference_variance_db2: Variance of the level difference.
        path_difference_variance_m2: Variance of the path difference.
        receiver_1_xy_m: Receiver 1 position `(x, y)` in metres.
        receiver_2_xy_m: Receiver 2 position `(x, y)` in metres.
        configuration: Domain side, singular tolerance and Jacobian step.

    Returns:
        Two-by-two position covariance in metres squared. It carries statistical
        uncertainty only; model error shows up in the reduced chi-square instead.
    """

    def forward(level_difference_db: float, difference_m: float) -> Float64Array:
        return compute_position_from_level_ratio(
            level_difference_db,
            difference_m,
            receiver_1_xy_m,
            receiver_2_xy_m,
            configuration.domain_side_sign,
        )

    step = configuration.jacobian_step
    derivative_level = (
        forward(geometric_level_difference_db + step, path_difference_m)
        - forward(geometric_level_difference_db - step, path_difference_m)
    ) / (2.0 * step)
    derivative_path = (
        forward(geometric_level_difference_db, path_difference_m + step)
        - forward(geometric_level_difference_db, path_difference_m - step)
    ) / (2.0 * step)

    jacobian = np.column_stack([derivative_level, derivative_path])
    input_covariance = np.diag(
        [geometric_level_difference_variance_db2, path_difference_variance_m2]
    )
    return jacobian @ input_covariance @ jacobian.T


def triangulate_from_level_ratio(
    geometric_level_difference_db: float,
    path_difference_m: float,
    geometric_level_difference_variance_db2: float,
    path_difference_variance_m2: float,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    configuration: TriangulationConfiguration,
) -> TriangulatedPosition:
    """Triangulates a position with its ranges, covariance and singular flag.

    Args:
        geometric_level_difference_db: Level difference in decibels.
        path_difference_m: Range difference in metres.
        geometric_level_difference_variance_db2: Variance of the level difference.
        path_difference_variance_m2: Variance of the path difference.
        receiver_1_xy_m: Receiver 1 position `(x, y)` in metres.
        receiver_2_xy_m: Receiver 2 position `(x, y)` in metres.
        configuration: Domain side, singular tolerance and Jacobian step.

    Returns:
        The triangulated position and everything needed to judge it.
    """
    range_1_m, range_2_m = compute_ranges_from_level_ratio(
        geometric_level_difference_db, path_difference_m
    )
    position_xy_m = triangulate_position_xy(
        range_1_m,
        range_2_m,
        receiver_1_xy_m,
        receiver_2_xy_m,
        configuration.domain_side_sign,
    )
    covariance = compute_position_covariance(
        geometric_level_difference_db,
        path_difference_m,
        geometric_level_difference_variance_db2,
        path_difference_variance_m2,
        receiver_1_xy_m,
        receiver_2_xy_m,
        configuration,
    )
    return TriangulatedPosition(
        position_xy_m=position_xy_m,
        position_covariance_m2=covariance,
        range_receiver_1_m=range_1_m,
        range_receiver_2_m=range_2_m,
        is_near_singular=is_near_perpendicular_bisector(
            geometric_level_difference_db, configuration.near_singular_tolerance
        ),
    )
