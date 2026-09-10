"""Receiver layouts in three dimensions, with the guards an array must pass.

The N-receiver generalisation of `receiver_placement.py`. Positions carry a
height here because a multi-source run solves for `(x, y)` while computing
ranges in 3D, so the microphone height belongs in the geometry rather than
being dropped.
"""

import numpy as np

from src.utils.array_types import Float64Array

REJECTION_SAMPLING_BATCH_FACTOR: int = 8
MAXIMUM_REJECTION_SAMPLING_ROUNDS: int = 1000
MINIMUM_RECEIVERS_FOR_COLLINEARITY: int = 3


def compute_minimum_pairwise_separation_m(
    receiver_positions_xyz: Float64Array,
) -> float:
    """Measures how close the two nearest receivers are.

    Args:
        receiver_positions_xyz: Positions, shape `(n_receivers, 3)`.

    Returns:
        The smallest distance between any two receivers, in metres. Infinite for
        a single receiver.
    """
    positions = np.atleast_2d(np.asarray(receiver_positions_xyz, dtype=np.float64))
    if positions.shape[0] < 2:
        return float("inf")
    differences = positions[:, None, :] - positions[None, :, :]
    distances_m = np.linalg.norm(differences, axis=2)
    distances_m[np.diag_indices(positions.shape[0])] = np.inf
    return float(np.min(distances_m))


def compute_collinearity_measure(receiver_positions_xyz: Float64Array) -> float:
    """Measures how close an array is to lying on a single line.

    One minus the ratio of the smaller to the larger singular value of the
    centred ground-plane positions: zero for a perfectly circular array, one for
    receivers on a line.

    A collinear array is the N-receiver form of the perpendicular-bisector
    degeneracy: a source and its mirror across the line produce identical delays
    at every receiver, so no estimator can separate them.

    Args:
        receiver_positions_xyz: Positions, shape `(n_receivers, 3)`.

    Returns:
        Collinearity in `[0, 1]`. Fewer than three receivers are always
        collinear and score one.
    """
    positions = np.atleast_2d(np.asarray(receiver_positions_xyz, dtype=np.float64))
    if positions.shape[0] < MINIMUM_RECEIVERS_FOR_COLLINEARITY:
        return 1.0
    centred_xy_m = positions[:, :2] - positions[:, :2].mean(axis=0)
    singular_values = np.linalg.svd(centred_xy_m, compute_uv=False)
    if singular_values[0] <= 0.0:
        return 1.0
    return float(1.0 - singular_values[-1] / singular_values[0])


def validate_receiver_layout(
    receiver_positions_xyz: Float64Array,
    minimum_separation_m: float,
    maximum_collinearity: float,
) -> Float64Array:
    """Checks a layout against the two guards and returns it unchanged.

    The collinearity guard is skipped below three receivers, where collinearity
    carries no information: two points always lie on a line, and the two-receiver
    case is handled by the bisector flag in `level_ratio_triangulation.py`
    instead.

    Args:
        receiver_positions_xyz: Positions, shape `(n_receivers, 3)`.
        minimum_separation_m: Smallest permitted distance between two receivers.
        maximum_collinearity: Largest permitted collinearity measure.

    Returns:
        The positions, unchanged.

    Raises:
        ValueError: If the receivers sit closer together than permitted, or the
            array is more collinear than permitted.
    """
    positions = np.atleast_2d(np.asarray(receiver_positions_xyz, dtype=np.float64))
    separation_m = compute_minimum_pairwise_separation_m(positions)
    if separation_m < minimum_separation_m:
        raise ValueError(
            f"the two nearest receivers are {separation_m:.2f} m apart, closer "
            f"than the required {minimum_separation_m:.2f} m"
        )
    if positions.shape[0] >= MINIMUM_RECEIVERS_FOR_COLLINEARITY:
        collinearity = compute_collinearity_measure(positions)
        if collinearity > maximum_collinearity:
            raise ValueError(
                f"the array is collinear to {collinearity:.4f}, above the "
                f"permitted {maximum_collinearity:.4f}; a source and its mirror "
                f"across that line produce identical delays at every receiver"
            )
    return positions


def build_ring_receiver_positions_xyz(
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    receiver_count: int,
    start_bearing_rad: float,
    height_m: float,
) -> Float64Array:
    """Spaces receivers evenly around a circle at a fixed height.

    Args:
        center_x_m: Ring centre along x, in metres.
        center_y_m: Ring centre along y, in metres.
        radius_m: Ring radius, in metres.
        receiver_count: Number of receivers to place.
        start_bearing_rad: Bearing of the first receiver, in radians.
        height_m: Receiver height above the ground plane, in metres.

    Returns:
        Positions of shape `(receiver_count, 3)`.

    Raises:
        ValueError: If the receiver count or the radius is not positive.
    """
    if receiver_count <= 0:
        raise ValueError("receiver count must be positive")
    if radius_m <= 0.0:
        raise ValueError("ring radius must be positive")

    bearings_rad = start_bearing_rad + np.linspace(
        0.0, 2.0 * np.pi, receiver_count, endpoint=False
    )
    return np.stack(
        (
            center_x_m + radius_m * np.cos(bearings_rad),
            center_y_m + radius_m * np.sin(bearings_rad),
            np.full(receiver_count, height_m, dtype=np.float64),
        ),
        axis=1,
    )


def build_grid_receiver_positions_xyz(
    origin_x_m: float,
    origin_y_m: float,
    extent_x_m: float,
    extent_y_m: float,
    spacing_m: float,
    height_m: float,
) -> Float64Array:
    """Tiles receivers on a regular grid at a fixed height.

    Args:
        origin_x_m: Lower-left corner along x, in metres.
        origin_y_m: Lower-left corner along y, in metres.
        extent_x_m: Width covered, in metres.
        extent_y_m: Height covered, in metres.
        spacing_m: Distance between adjacent receivers, in metres.
        height_m: Receiver height above the ground plane, in metres.

    Returns:
        Positions of shape `(n_receivers, 3)`, x varying fastest.

    Raises:
        ValueError: If the spacing is not positive.
    """
    if spacing_m <= 0.0:
        raise ValueError("receiver spacing must be positive")

    half_spacing_m = 0.5 * spacing_m
    x_m = np.arange(origin_x_m, origin_x_m + extent_x_m + half_spacing_m, spacing_m)
    y_m = np.arange(origin_y_m, origin_y_m + extent_y_m + half_spacing_m, spacing_m)
    grid_x_m, grid_y_m = np.meshgrid(x_m, y_m)
    return np.stack(
        (
            grid_x_m.ravel(),
            grid_y_m.ravel(),
            np.full(grid_x_m.size, height_m, dtype=np.float64),
        ),
        axis=1,
    )


def build_random_receiver_positions_xyz(
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    receiver_count: int,
    minimum_separation_m: float,
    height_m: float,
    seed: int,
) -> Float64Array:
    """Draws receivers inside a disc, rejecting any that crowd an accepted one.

    Poisson-disc rejection rather than plain uniform sampling: two receivers a
    few centimetres apart carry one measurement between them and inflate the
    pair count without adding information.

    Args:
        center_x_m: Disc centre along x, in metres.
        center_y_m: Disc centre along y, in metres.
        radius_m: Disc radius, in metres.
        receiver_count: Number of receivers to place.
        minimum_separation_m: Smallest permitted distance between two receivers.
        height_m: Receiver height above the ground plane, in metres.
        seed: Seed making the layout reproducible.

    Returns:
        Positions of shape `(receiver_count, 3)`.

    Raises:
        ValueError: If the receiver count or the radius is not positive, or the
            requested separation cannot be met inside the disc.
    """
    if receiver_count <= 0:
        raise ValueError("receiver count must be positive")
    if radius_m <= 0.0:
        raise ValueError("sampling radius must be positive")

    generator = np.random.default_rng(seed)
    batch_size = REJECTION_SAMPLING_BATCH_FACTOR * receiver_count
    accepted_xy_m = np.empty((0, 2), dtype=np.float64)
    for _ in range(MAXIMUM_REJECTION_SAMPLING_ROUNDS):
        candidates_xy_m = generator.uniform(-radius_m, radius_m, size=(batch_size, 2))
        candidates_xy_m = candidates_xy_m[
            np.sum(candidates_xy_m**2, axis=1) <= radius_m**2
        ]
        for candidate_xy_m in candidates_xy_m:
            if accepted_xy_m.shape[0] > 0 and np.any(
                np.linalg.norm(accepted_xy_m - candidate_xy_m, axis=1)
                < minimum_separation_m
            ):
                continue
            accepted_xy_m = np.vstack((accepted_xy_m, candidate_xy_m))
            if accepted_xy_m.shape[0] == receiver_count:
                centre_xy_m = np.array([center_x_m, center_y_m], dtype=np.float64)
                return np.column_stack(
                    (
                        accepted_xy_m + centre_xy_m,
                        np.full(receiver_count, height_m, dtype=np.float64),
                    )
                )
    raise ValueError(
        f"could not place {receiver_count} receivers at least "
        f"{minimum_separation_m:.2f} m apart inside a disc of radius "
        f"{radius_m:.2f} m"
    )


def lift_positions_to_height_xyz(
    positions_xy_m: Float64Array, height_m: float
) -> Float64Array:
    """Gives ground-plane positions a fixed height.

    Args:
        positions_xy_m: Positions, shape `(n_positions, 2)`.
        height_m: Height above the ground plane, in metres.

    Returns:
        Positions of shape `(n_positions, 3)`.
    """
    positions = np.atleast_2d(np.asarray(positions_xy_m, dtype=np.float64))
    return np.column_stack(
        (positions, np.full(positions.shape[0], height_m, dtype=np.float64))
    )
