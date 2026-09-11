"""What the fire actually did, reduced to the quantities an estimator claims.

Reads the fire state directly, which is why it lives here and not in
`inverse/`. Nothing in `inverse/` may import it, tests included — a test that
needs a front builds a synthetic map instead.

Every definition is fixed here so it is not re-litigated per estimator. In
particular the truth and the estimate are reduced the same way: the true
semi-axes come from the active cells' own covariance scaled by the same shape
factor the extent stage applies to a map, so the two are comparable without
either side being given an advantage the other does not have.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.utils.array_types import Float64Array

MINIMUM_HULL_POINT_COUNT: int = 3


@dataclass(frozen=True)
class FireFrontTruth:
    """One observation of the fire, as the quantities to be estimated.

    Attributes:
        time_s: Simulated time of the observation.
        active_cell_positions_xy_m: Radiating cells, shape `(n_active, 2)`.
        ignition_position_xy_m: Where the fire started, shape `(2,)`.
        front_centroid_xy_m: Mean position of the active cells, shape `(2,)`.
        head_position_xy_m: Active cell furthest along the spread direction.
        back_position_xy_m: Active cell furthest against it.
        equivalent_radius_m: Radius of a disc with the active cells' hull area.
        bearing_rad: Direction from ignition to the head, in radians.
        principal_semi_axes_m: Major and minor semi-axes, shape `(2,)`.
        head_distance_m: Distance from ignition to the head, along the bearing.
        back_distance_m: Distance from ignition to the back, along the bearing.
    """

    time_s: float
    active_cell_positions_xy_m: Float64Array
    ignition_position_xy_m: Float64Array
    front_centroid_xy_m: Float64Array
    head_position_xy_m: Float64Array
    back_position_xy_m: Float64Array
    equivalent_radius_m: float
    bearing_rad: float
    principal_semi_axes_m: Float64Array
    head_distance_m: float
    back_distance_m: float


def compute_hull_area_m2(positions_xy_m: Float64Array, cell_area_m2: float) -> float:
    """Area the active cells enclose.

    Falls back to counting cells when there are too few points for a hull, or
    when they are collinear, which happens in the first seconds while the fire
    is one or two cells across.

    Args:
        positions_xy_m: Active cells, shape `(n_active, 2)`.
        cell_area_m2: Area of one mesh cell, in square metres.

    Returns:
        Enclosed area in square metres.
    """
    positions = np.atleast_2d(np.asarray(positions_xy_m, dtype=np.float64))
    if positions.shape[0] < MINIMUM_HULL_POINT_COUNT:
        return float(positions.shape[0] * cell_area_m2)
    try:
        return float(ConvexHull(positions).volume)
    except QhullError:
        return float(positions.shape[0] * cell_area_m2)


def compute_principal_semi_axes_m(
    positions_xy_m: Float64Array, shape_factor: float
) -> Float64Array:
    """Semi-axes of the active cells, by the same rule the estimate uses.

    Args:
        positions_xy_m: Active cells, shape `(n_active, 2)`.
        shape_factor: Multiplier from the root of an eigenvalue to a semi-axis.

    Returns:
        `(major_m, minor_m)`, shape `(2,)`, descending.
    """
    positions = np.atleast_2d(np.asarray(positions_xy_m, dtype=np.float64))
    if positions.shape[0] < 2:
        return np.zeros(2, dtype=np.float64)
    covariance_m2 = np.cov(positions.T, bias=True)
    eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance_m2))
    return np.asarray(
        shape_factor * np.sqrt(np.clip(eigenvalues, 0.0, None))[::-1], dtype=np.float64
    )


def extract_fire_front_truth(
    time_s: float,
    active_cell_positions_xy_m: Float64Array,
    ignition_position_xy_m: Float64Array,
    cell_area_m2: float,
    shape_factor: float,
) -> FireFrontTruth:
    """Reduces one fire observation to the quantities an estimator claims.

    The spread direction is taken from ignition to the front's centroid rather
    than to its brightest cell, because a centroid is defined whatever the
    front's shape and does not jump between cells. The head and back are then
    the active cells furthest along and against that direction.

    Args:
        time_s: Simulated time of the observation.
        active_cell_positions_xy_m: Radiating cells, shape `(n_active, 2)`.
        ignition_position_xy_m: Where the fire started, shape `(2,)`.
        cell_area_m2: Area of one mesh cell, in square metres.
        shape_factor: Multiplier from the root of an eigenvalue to a semi-axis.

    Returns:
        The observation's truth.

    Raises:
        ValueError: If no cell is active.
    """
    positions = np.atleast_2d(np.asarray(active_cell_positions_xy_m, dtype=np.float64))
    if positions.shape[0] == 0:
        raise ValueError("a fire observation with no active cell has no front")
    ignition_xy_m = np.asarray(ignition_position_xy_m, dtype=np.float64)
    centroid_xy_m = np.mean(positions, axis=0)

    spread_vector = centroid_xy_m - ignition_xy_m
    spread_length_m = float(np.linalg.norm(spread_vector))
    direction = (
        spread_vector / spread_length_m
        if spread_length_m > 0.0
        else np.array([1.0, 0.0])
    )
    projections_m = (positions - ignition_xy_m) @ direction
    head_position_xy_m = positions[int(np.argmax(projections_m))]
    back_position_xy_m = positions[int(np.argmin(projections_m))]

    head_vector = head_position_xy_m - ignition_xy_m
    return FireFrontTruth(
        time_s=time_s,
        active_cell_positions_xy_m=positions,
        ignition_position_xy_m=ignition_xy_m,
        front_centroid_xy_m=centroid_xy_m,
        head_position_xy_m=head_position_xy_m,
        back_position_xy_m=back_position_xy_m,
        equivalent_radius_m=float(
            np.sqrt(compute_hull_area_m2(positions, cell_area_m2) / np.pi)
        ),
        bearing_rad=float(np.arctan2(head_vector[1], head_vector[0])),
        principal_semi_axes_m=compute_principal_semi_axes_m(positions, shape_factor),
        head_distance_m=float(np.max(projections_m)),
        back_distance_m=float(np.min(projections_m)),
    )


def compute_true_distance_by_angle_m(
    active_cell_positions_xy_m: Float64Array,
    origin_xy_m: Float64Array,
    angles_rad: Float64Array,
    angular_tolerance_rad: float,
) -> Float64Array:
    """Outer radius of the active cells in each direction.

    The front is a band of cells rather than a curve, so its outer edge in a
    direction is the furthest active cell within a narrow wedge about that
    direction. A wedge containing no active cell reports `nan`, which is how
    the front's angular coverage comes back as a measurement rather than being
    assumed to be the full circle.

    Args:
        active_cell_positions_xy_m: Radiating cells, shape `(n_active, 2)`.
        origin_xy_m: Where rays start, shape `(2,)`.
        angles_rad: Directions to evaluate, shape `(n_angles,)`.
        angular_tolerance_rad: Half-width of the wedge about each direction.

    Returns:
        Distance at each angle, shape `(n_angles,)`, `nan` where uncovered.

    Raises:
        ValueError: If the tolerance is not positive.
    """
    if angular_tolerance_rad <= 0.0:
        raise ValueError("the angular tolerance must be positive")
    positions = np.atleast_2d(np.asarray(active_cell_positions_xy_m, dtype=np.float64))
    angles = np.asarray(angles_rad, dtype=np.float64)
    if positions.shape[0] == 0:
        return np.full(angles.size, np.nan, dtype=np.float64)

    offsets_m = positions - np.asarray(origin_xy_m, dtype=np.float64)
    cell_radii_m = np.linalg.norm(offsets_m, axis=1)
    cell_angles_rad = np.arctan2(offsets_m[:, 1], offsets_m[:, 0])

    separation_rad = np.abs(
        np.arctan2(
            np.sin(cell_angles_rad[None, :] - angles[:, None]),
            np.cos(cell_angles_rad[None, :] - angles[:, None]),
        )
    )
    inside = separation_rad <= angular_tolerance_rad
    distances_m = np.where(
        np.any(inside, axis=1),
        np.max(np.where(inside, cell_radii_m[None, :], -np.inf), axis=1),
        np.nan,
    )
    return np.asarray(distances_m, dtype=np.float64)


def compute_true_bearing_from_sequence_rad(truths: list[FireFrontTruth]) -> float:
    """One bearing for a whole run, from ignition to the last head position.

    A per-observation bearing is noisy while the fire is a few cells across, so
    a run-level comparison uses the direction the fire actually ended up
    travelling rather than an average of early guesses.

    Args:
        truths: The run's observations, in time order.

    Returns:
        The bearing in radians.

    Raises:
        ValueError: If the sequence is empty.
    """
    if not truths:
        raise ValueError("an empty sequence has no bearing")
    last = truths[-1]
    vector = last.head_position_xy_m - last.ignition_position_xy_m
    return float(np.arctan2(vector[1], vector[0]))
