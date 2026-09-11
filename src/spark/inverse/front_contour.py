"""The front's outer edge, as a distance in every direction it covers.

Extraction is radial rather than by marching squares, and that is a choice the
scene forces rather than a shortcut. The radiating front of a fire that burns
out behind itself is an open arc — measured on `configs/f1` it spans about
ninety degrees with a two-hundred-and-seventy degree gap at every observation —
so there is no closed curve to march around. Sampling outward along each
bearing and taking the outermost crossing handles that directly: a direction
the front does not cover simply has no crossing, and reports one.

That also makes the angular coverage an output. A contour extractor that
returned a closed curve regardless would have to invent the silent three
quarters, and every metric taken against it would be measuring the invention.
"""

import numpy as np

from src.spark.inverse.front_radial_profile import sample_map_bilinear
from src.utils.array_types import Float64Array

DENSITY_FLOOR: float = 1e-12


def compute_contour_distance_by_angle(
    source_density: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    origin_xy_m: Float64Array,
    angles_rad: Float64Array,
    level_fraction: float,
    maximum_radius_m: float,
    radius_step_m: float,
) -> Float64Array:
    """Outermost radius at which the density falls through a level, per angle.

    Args:
        source_density: Deconvolved density, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        origin_xy_m: Where rays start, shape `(2,)`.
        angles_rad: Directions to sample, shape `(n_angles,)`.
        level_fraction: Level as a fraction of the density's global maximum.
        maximum_radius_m: Furthest radius sampled.
        radius_step_m: Spacing along each ray, in metres.

    Returns:
        Distance at each angle, shape `(n_angles,)`, with `nan` where the ray
        never rises above the level — which is how an uncovered direction
        reports itself.

    Raises:
        ValueError: If the level fraction is outside `(0, 1]`, or the sampling
            parameters are not positive.
    """
    if not 0.0 < level_fraction <= 1.0:
        raise ValueError("the contour level fraction must lie in (0, 1]")
    if maximum_radius_m <= 0.0 or radius_step_m <= 0.0:
        raise ValueError("the maximum radius and step must both be positive")

    density = np.asarray(source_density, dtype=np.float64)
    peak = float(np.max(density))
    if peak <= DENSITY_FLOOR:
        return np.full(len(angles_rad), np.nan, dtype=np.float64)
    level = level_fraction * peak

    radii_m = np.arange(radius_step_m, maximum_radius_m + radius_step_m, radius_step_m)
    origin = np.asarray(origin_xy_m, dtype=np.float64)
    distances_m = np.full(len(angles_rad), np.nan, dtype=np.float64)
    for angle_index, angle_rad in enumerate(np.asarray(angles_rad, dtype=np.float64)):
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        samples = sample_map_bilinear(
            density,
            candidate_positions_xyz_m,
            origin + np.outer(radii_m, direction),
        )
        above = np.flatnonzero(samples >= level)
        if above.size == 0:
            continue
        outermost = int(above[-1])
        distances_m[angle_index] = _refine_crossing_m(
            radii_m, samples, outermost, level
        )
    return distances_m


def extract_front_contour(
    source_density: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    origin_xy_m: Float64Array,
    angles_rad: Float64Array,
    level_fraction: float,
    maximum_radius_m: float,
    radius_step_m: float,
) -> Float64Array:
    """The contour as points, for the directions it actually covers.

    Args:
        source_density: Deconvolved density, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        origin_xy_m: Where rays start, shape `(2,)`.
        angles_rad: Directions to sample, shape `(n_angles,)`.
        level_fraction: Level as a fraction of the density's global maximum.
        maximum_radius_m: Furthest radius sampled.
        radius_step_m: Spacing along each ray, in metres.

    Returns:
        Contour points of shape `(n_covered, 2)`, in angle order. Directions
        the front does not cover are absent rather than interpolated across.
    """
    distances_m = compute_contour_distance_by_angle(
        source_density,
        candidate_positions_xyz_m,
        origin_xy_m,
        angles_rad,
        level_fraction,
        maximum_radius_m,
        radius_step_m,
    )
    angles = np.asarray(angles_rad, dtype=np.float64)
    covered = np.isfinite(distances_m)
    if not bool(np.any(covered)):
        return np.empty((0, 2), dtype=np.float64)
    origin = np.asarray(origin_xy_m, dtype=np.float64)
    return np.asarray(
        origin
        + np.stack(
            (
                distances_m[covered] * np.cos(angles[covered]),
                distances_m[covered] * np.sin(angles[covered]),
            ),
            axis=1,
        ),
        dtype=np.float64,
    )


def compute_angular_coverage_fraction(distances_m: Float64Array) -> float:
    """Share of the swept directions in which a front was found.

    Args:
        distances_m: Distance per angle, with `nan` where uncovered.

    Returns:
        Fraction of angles that carry a distance.
    """
    values = np.asarray(distances_m, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(values)))


def _refine_crossing_m(
    radii_m: Float64Array,
    samples: Float64Array,
    outermost_index: int,
    level: float,
) -> float:
    """Linearly interpolates where the profile last crossed the level.

    Args:
        radii_m: Radii sampled along the ray.
        samples: Density at each radius.
        outermost_index: Last index at or above the level.
        level: The level being crossed.

    Returns:
        The crossing radius, in metres.
    """
    if outermost_index + 1 >= samples.size:
        return float(radii_m[outermost_index])
    inside = float(samples[outermost_index])
    outside = float(samples[outermost_index + 1])
    if inside <= outside:
        return float(radii_m[outermost_index])
    weight = (inside - level) / (inside - outside)
    step_m = float(radii_m[outermost_index + 1] - radii_m[outermost_index])
    return float(radii_m[outermost_index] + float(np.clip(weight, 0.0, 1.0)) * step_m)
