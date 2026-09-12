"""The front's outer edge, as a distance in every direction it covers.

Extraction is radial rather than by marching squares, and that is a choice the
scene forces rather than a shortcut. The radiating front of a fire that burns
out behind itself is an open arc — measured on `configs/f1` it spans about
ninety degrees with a two-hundred-and-seventy degree gap at every observation —
so there is no closed curve to march around. Sampling outward along each
bearing and taking the outermost crossing handles that directly: a direction
the front does not cover simply has no crossing, and reports one.

A crossing on its own is not enough to call a direction covered, and that is
what the angular support gate below is for. The level is a fraction of the
density's *global* peak, so a direction whose only claim is the skirt of a
bright arc somewhere else still crosses it, close in, and comes back with a
radius belonging to that arc rather than to itself. Support is measured per
direction instead: a direction carries a front only when the wedge mass it
holds is a fair share of the best direction's, which is a statement about that
direction alone and cannot be borrowed from a neighbour.

The level has the same problem and takes the same answer. Referred to the
global peak alone it keeps only the brightest stretch of a front, which costs
nothing on `configs/f1` — an arc of fairly even brightness — and almost
everything on `configs/f2`, where a probabilistic spread model produces a
closed but ragged front and the contour claimed 0.200 of the circle against a
true 0.994. So there are two levels: a floor against the global peak, which
keeps a direction carrying only noise from claiming anything, and a level
against each ray's own peak, which lets a dim stretch report the front it has.

That is what makes the angular coverage an output rather than an assumption. A
contour extractor that returned a closed curve regardless would have to invent
the silent three quarters, and every metric taken against it would be measuring
the invention — and one that could only ever return an open arc would be making
the opposite mistake on a fire that really does surround its ignition point.
"""

import numpy as np

from src.spark.inverse.front_radial_profile import sample_map_bilinear
from src.utils.array_types import Float64Array

DENSITY_FLOOR: float = 1e-12


def compute_angular_support(
    source_density: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    origin_xy_m: Float64Array,
    angles_rad: Float64Array,
    maximum_radius_m: float,
    radius_step_m: float,
) -> Float64Array:
    """Wedge mass the density places in each direction, against the best one.

    The radius weight is what makes this a mass rather than a line integral. A
    wedge's area grows with its radius, so ground far out counts for what it is
    worth; without the weight a bright speck beside the origin would outvote a
    whole arc twelve metres away.

    Args:
        source_density: Deconvolved density, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        origin_xy_m: Where rays start, shape `(2,)`.
        angles_rad: Directions to sample, shape `(n_angles,)`.
        maximum_radius_m: Furthest radius sampled.
        radius_step_m: Spacing along each ray, in metres.

    Returns:
        Support per angle, shape `(n_angles,)`, scaled so the best direction is
        one. All zero when the density carries nothing.

    Raises:
        ValueError: If the sampling parameters are not positive.
    """
    radii_m = _build_ray_radii_m(maximum_radius_m, radius_step_m)
    samples = _sample_rays(
        source_density, candidate_positions_xyz_m, origin_xy_m, angles_rad, radii_m
    )
    return _compute_support_from_samples(samples, radii_m)


def compute_contour_distance_by_angle(
    source_density: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    origin_xy_m: Float64Array,
    angles_rad: Float64Array,
    level_fraction: float,
    local_level_fraction: float,
    support_fraction: float,
    maximum_radius_m: float,
    radius_step_m: float,
) -> Float64Array:
    """Outermost radius at which the density falls through a level, per angle.

    Two levels, and a direction must clear the higher of them. The first is a
    fraction of the density's global peak and is a floor: it is what keeps a
    direction carrying nothing but noise from claiming a front. The second is a
    fraction of that ray's *own* peak, and it is what lets a front be found in
    a direction the fire happens to burn dimly. A front of uneven brightness
    otherwise loses its dim stretches entirely — measured on `configs/f2`, a
    global level alone claimed a fifth of the circle where the fire covered
    essentially all of it. Set the local fraction to zero for a purely global
    level.

    Args:
        source_density: Deconvolved density, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        origin_xy_m: Where rays start, shape `(2,)`.
        angles_rad: Directions to sample, shape `(n_angles,)`.
        level_fraction: Floor level, as a fraction of the density's global
            maximum.
        local_level_fraction: Level as a fraction of each ray's own maximum.
            Zero leaves the floor as the only level.
        support_fraction: Least wedge mass, as a fraction of the best
            direction's, a direction must hold to carry a front at all. Zero
            asks for no gate and leaves every crossing standing.
        maximum_radius_m: Furthest radius sampled.
        radius_step_m: Spacing along each ray, in metres.

    Returns:
        Distance at each angle, shape `(n_angles,)`, with `nan` where the ray
        never rises above its level or holds too little mass to be believed —
        which is how an uncovered direction reports itself.

    Raises:
        ValueError: If either level fraction is out of range, the support
            fraction is outside `[0, 1]`, or the sampling parameters are not
            positive.
    """
    if not 0.0 < level_fraction <= 1.0:
        raise ValueError("the contour level fraction must lie in (0, 1]")
    if not 0.0 <= local_level_fraction <= 1.0:
        raise ValueError("the local contour level fraction must lie in [0, 1]")
    if not 0.0 <= support_fraction <= 1.0:
        raise ValueError("the angular support fraction must lie in [0, 1]")

    radii_m = _build_ray_radii_m(maximum_radius_m, radius_step_m)
    angles = np.asarray(angles_rad, dtype=np.float64)
    density = np.asarray(source_density, dtype=np.float64)
    peak = float(np.max(density)) if density.size else 0.0
    if peak <= DENSITY_FLOOR:
        return np.full(angles.size, np.nan, dtype=np.float64)
    floor_level = level_fraction * peak

    samples = _sample_rays(
        density, candidate_positions_xyz_m, origin_xy_m, angles, radii_m
    )
    supported = _compute_support_from_samples(samples, radii_m) >= support_fraction
    ray_levels = np.maximum(floor_level, local_level_fraction * np.max(samples, axis=1))

    distances_m = np.full(angles.size, np.nan, dtype=np.float64)
    for angle_index in range(angles.size):
        if not bool(supported[angle_index]):
            continue
        ray = samples[angle_index]
        level = float(ray_levels[angle_index])
        above = np.flatnonzero(ray >= level)
        if above.size == 0:
            continue
        distances_m[angle_index] = _refine_crossing_m(
            radii_m, ray, int(above[-1]), level
        )
    return distances_m


def extract_front_contour(
    source_density: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    origin_xy_m: Float64Array,
    angles_rad: Float64Array,
    level_fraction: float,
    local_level_fraction: float,
    support_fraction: float,
    maximum_radius_m: float,
    radius_step_m: float,
) -> Float64Array:
    """The contour as points, for the directions it actually covers.

    Args:
        source_density: Deconvolved density, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        origin_xy_m: Where rays start, shape `(2,)`.
        angles_rad: Directions to sample, shape `(n_angles,)`.
        level_fraction: Floor level, as a fraction of the global maximum.
        local_level_fraction: Level as a fraction of each ray's own maximum.
        support_fraction: Least wedge mass a direction must hold to carry a
            front, as a fraction of the best direction's.
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
        local_level_fraction,
        support_fraction,
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


def _build_ray_radii_m(maximum_radius_m: float, radius_step_m: float) -> Float64Array:
    """Radii every ray is sampled at.

    Args:
        maximum_radius_m: Furthest radius sampled.
        radius_step_m: Spacing along each ray, in metres.

    Returns:
        Ascending radii, shape `(n_radii,)`.

    Raises:
        ValueError: If either parameter is not positive.
    """
    if maximum_radius_m <= 0.0 or radius_step_m <= 0.0:
        raise ValueError("the maximum radius and step must both be positive")
    return np.arange(
        radius_step_m, maximum_radius_m + radius_step_m, radius_step_m, dtype=np.float64
    )


def _sample_rays(
    source_density: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    origin_xy_m: Float64Array,
    angles_rad: Float64Array,
    radii_m: Float64Array,
) -> Float64Array:
    """Density along every ray, in one pass.

    Args:
        source_density: Deconvolved density, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        origin_xy_m: Where rays start, shape `(2,)`.
        angles_rad: Directions to sample, shape `(n_angles,)`.
        radii_m: Radii along each ray, shape `(n_radii,)`.

    Returns:
        Sampled density of shape `(n_angles, n_radii)`.
    """
    angles = np.asarray(angles_rad, dtype=np.float64)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    query_positions_xy_m = (
        np.asarray(origin_xy_m, dtype=np.float64)
        + radii_m[None, :, None] * directions[:, None, :]
    ).reshape(-1, 2)
    samples = sample_map_bilinear(
        source_density, candidate_positions_xyz_m, query_positions_xy_m
    )
    return np.asarray(samples.reshape(angles.size, radii_m.size), dtype=np.float64)


def _compute_support_from_samples(
    samples: Float64Array, radii_m: Float64Array
) -> Float64Array:
    """Wedge mass per direction, scaled so the best direction is one.

    Args:
        samples: Sampled density of shape `(n_angles, n_radii)`.
        radii_m: Radii along each ray, shape `(n_radii,)`.

    Returns:
        Support per angle, shape `(n_angles,)`.
    """
    mass = np.sum(samples * radii_m[None, :], axis=1)
    peak = float(np.max(mass)) if mass.size else 0.0
    if peak <= DENSITY_FLOOR:
        return np.zeros(mass.size, dtype=np.float64)
    return np.asarray(mass / peak, dtype=np.float64)


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
