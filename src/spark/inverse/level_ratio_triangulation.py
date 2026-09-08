from dataclasses import dataclass

import numpy as np

from src.config.simulation_configuration import TriangulationConfiguration
from src.utils.array_types import Float64Array

LEVEL_RATIO_FLOOR: float = 1e-9


@dataclass(frozen=True)
class TriangulatedPosition:
    position_xy_m: Float64Array
    position_covariance_m2: Float64Array
    range_receiver_1_m: float
    range_receiver_2_m: float
    is_near_singular: bool


def compute_level_ratio(geometric_level_difference_db: float) -> float:
    return float(10.0 ** (geometric_level_difference_db / 20.0))


def is_near_perpendicular_bisector(
    geometric_level_difference_db: float,
    near_singular_tolerance: float,
) -> bool:
    return (
        abs(compute_level_ratio(geometric_level_difference_db) - 1.0)
        < near_singular_tolerance
    )


def compute_ranges_from_level_ratio(
    geometric_level_difference_db: float,
    path_difference_m: float,
) -> tuple[float, float]:
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

    return (
        receiver_1
        + local_x_m * unit_along_baseline
        + domain_side_sign * local_y_m * unit_across_baseline
    )


def compute_position_from_level_ratio(
    geometric_level_difference_db: float,
    path_difference_m: float,
    receiver_1_xy_m: Float64Array,
    receiver_2_xy_m: Float64Array,
    domain_side_sign: float,
) -> Float64Array:
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
