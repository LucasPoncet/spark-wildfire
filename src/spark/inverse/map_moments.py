"""Mass-weighted moments of a prepared steered response power map.

The map is treated as a density over the candidate grid. Its zeroth moment is
how much energy the frame carries, its first is where that energy sits, its
second is how far it spreads and along which axes, and its third along a chosen
direction is how lopsided it is — which is what a head fire burning brighter
than its back looks like in a map that cannot resolve either.

Every quantity is a single vectorised expression over the grid. Nothing here
knows the map came from a fire, or from a rendering rather than from an
analytic point spread function; the extent stage differences two covariances
computed by this module, so both sides must be measured the same way.
"""

from dataclasses import dataclass

import numpy as np

from src.spark.inverse.map_normalization import prepare_map_for_moments
from src.utils.array_types import Float64Array

VARIANCE_FLOOR_M2: float = 1e-12


@dataclass(frozen=True)
class MapMoments:
    """What one prepared frame's mass distribution looks like.

    The directional third moment the plan lists as a field is a free function
    here instead. A callable on a frozen dataclass cannot be serialised into a
    metrics record and would be the only field in this package that is not
    data, so `compute_directional_third_moment` takes the map explicitly.

    Attributes:
        total_mass: Sum of the prepared map, one by construction unless the
            frame was empty.
        centroid_xy_m: Mass-weighted mean position, shape `(2,)`.
        covariance_xy_m2: Second central moment, shape `(2, 2)`.
        principal_axes: Unit eigenvectors as columns, major first, shape
            `(2, 2)`.
        principal_variances_m2: Eigenvalues in descending order, shape `(2,)`.
    """

    total_mass: float
    centroid_xy_m: Float64Array
    covariance_xy_m2: Float64Array
    principal_axes: Float64Array
    principal_variances_m2: Float64Array


def compute_map_moments(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    background_percentile: float,
) -> MapMoments:
    """Reduces one frame to its mass, centroid and second central moment.

    Args:
        map_values: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        background_percentile: Percentile of the frame treated as background.

    Returns:
        The frame's moments. An empty frame returns zero mass, the grid centre
        as its centroid and a zero covariance rather than raising, so a
        sequence containing a silent frame can still be assembled.
    """
    weights = prepare_map_for_moments(map_values, background_percentile)
    positions_xy_m = np.atleast_2d(
        np.asarray(candidate_positions_xyz_m, dtype=np.float64)
    )[:, :2]
    total_mass = float(np.sum(weights))
    if total_mass <= 0.0:
        return MapMoments(
            total_mass=0.0,
            centroid_xy_m=np.mean(positions_xy_m, axis=0),
            covariance_xy_m2=np.zeros((2, 2), dtype=np.float64),
            principal_axes=np.eye(2, dtype=np.float64),
            principal_variances_m2=np.zeros(2, dtype=np.float64),
        )

    centroid_xy_m = np.asarray(weights @ positions_xy_m, dtype=np.float64)
    offsets_m = positions_xy_m - centroid_xy_m
    covariance_xy_m2 = np.asarray(
        (offsets_m * weights[:, None]).T @ offsets_m, dtype=np.float64
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xy_m2)
    order = np.argsort(eigenvalues)[::-1]
    return MapMoments(
        total_mass=total_mass,
        centroid_xy_m=centroid_xy_m,
        covariance_xy_m2=covariance_xy_m2,
        principal_axes=np.asarray(eigenvectors[:, order], dtype=np.float64),
        principal_variances_m2=np.asarray(eigenvalues[order], dtype=np.float64),
    )


def compute_directional_third_moment(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    centroid_xy_m: Float64Array,
    direction_xy: Float64Array,
    background_percentile: float,
) -> float:
    """Third central moment of the mass along one direction.

    Args:
        map_values: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        centroid_xy_m: Centroid the offsets are taken from, shape `(2,)`.
        direction_xy: Direction to project onto, shape `(2,)`, normalised here.
        background_percentile: Percentile of the frame treated as background.

    Returns:
        The third moment in metres cubed. Positive means the mass leans along
        the direction given.

    Raises:
        ValueError: If the direction has zero length.
    """
    weights = prepare_map_for_moments(map_values, background_percentile)
    projections_m = _project_offsets_m(
        candidate_positions_xyz_m, centroid_xy_m, direction_xy
    )
    total_mass = float(np.sum(weights))
    if total_mass <= 0.0:
        return 0.0
    return float(weights @ projections_m**3 / total_mass)


def compute_directional_skewness(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    centroid_xy_m: Float64Array,
    direction_xy: Float64Array,
    background_percentile: float,
) -> float:
    """Third moment along a direction, made dimensionless by the second.

    Args:
        map_values: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        centroid_xy_m: Centroid the offsets are taken from, shape `(2,)`.
        direction_xy: Direction to project onto, shape `(2,)`.
        background_percentile: Percentile of the frame treated as background.

    Returns:
        The skewness along that direction, dimensionless. Zero when the mass
        has no spread along it, rather than a division by zero.
    """
    weights = prepare_map_for_moments(map_values, background_percentile)
    projections_m = _project_offsets_m(
        candidate_positions_xyz_m, centroid_xy_m, direction_xy
    )
    total_mass = float(np.sum(weights))
    if total_mass <= 0.0:
        return 0.0
    variance_m2 = float(weights @ projections_m**2 / total_mass)
    if variance_m2 <= VARIANCE_FLOOR_M2:
        return 0.0
    third_moment_m3 = float(weights @ projections_m**3 / total_mass)
    return float(third_moment_m3 / variance_m2**1.5)


def compute_skewness_over_directions(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    centroid_xy_m: Float64Array,
    direction_count: int,
    background_percentile: float,
) -> tuple[Float64Array, Float64Array]:
    """Sweeps the directional skewness over a full turn in one pass.

    The map is prepared once and every direction is evaluated against the same
    weights, so this is one matrix product rather than `direction_count` calls.

    Args:
        map_values: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        centroid_xy_m: Centroid the offsets are taken from, shape `(2,)`.
        direction_count: Number of directions to sample over a full turn.
        background_percentile: Percentile of the frame treated as background.

    Returns:
        `(angles_rad, skewness)`, both of shape `(direction_count,)`, with the
        angles ascending from zero.

    Raises:
        ValueError: If fewer than three directions are requested.
    """
    if direction_count < 3:
        raise ValueError("a skewness sweep needs at least three directions")
    weights = prepare_map_for_moments(map_values, background_percentile)
    positions_xy_m = np.atleast_2d(
        np.asarray(candidate_positions_xyz_m, dtype=np.float64)
    )[:, :2]
    offsets_m = positions_xy_m - np.asarray(centroid_xy_m, dtype=np.float64)

    angles_rad = np.linspace(0.0, 2.0 * np.pi, direction_count, endpoint=False)
    directions_xy = np.stack((np.cos(angles_rad), np.sin(angles_rad)), axis=1)
    total_mass = float(np.sum(weights))
    if total_mass <= 0.0:
        return angles_rad, np.zeros(direction_count, dtype=np.float64)

    projections_m = offsets_m @ directions_xy.T
    variances_m2 = weights @ projections_m**2 / total_mass
    third_moments_m3 = weights @ projections_m**3 / total_mass
    skewness = np.where(
        variances_m2 > VARIANCE_FLOOR_M2,
        third_moments_m3 / np.maximum(variances_m2, VARIANCE_FLOOR_M2) ** 1.5,
        0.0,
    )
    return angles_rad, np.asarray(skewness, dtype=np.float64)


def _project_offsets_m(
    candidate_positions_xyz_m: Float64Array,
    centroid_xy_m: Float64Array,
    direction_xy: Float64Array,
) -> Float64Array:
    """Projects every cell's offset from the centroid onto a unit direction.

    Args:
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        centroid_xy_m: Centroid the offsets are taken from, shape `(2,)`.
        direction_xy: Direction to project onto, shape `(2,)`.

    Returns:
        Signed distances along the direction, shape `(n_cells,)`.

    Raises:
        ValueError: If the direction has zero length.
    """
    direction = np.asarray(direction_xy, dtype=np.float64)
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 0.0:
        raise ValueError("a projection direction must have non-zero length")
    positions_xy_m = np.atleast_2d(
        np.asarray(candidate_positions_xyz_m, dtype=np.float64)
    )[:, :2]
    offsets_m = positions_xy_m - np.asarray(centroid_xy_m, dtype=np.float64)
    return np.asarray(offsets_m @ (direction / direction_length), dtype=np.float64)
