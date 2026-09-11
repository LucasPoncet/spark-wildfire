"""Which way the fire is going, from a sequence of imaging maps.

Four estimators, chosen because they fail differently rather than for
redundancy. Centroid drift needs the front to have moved and says nothing about
a single frame. Centroid offset needs the ignition point but no motion, so it
yields a bearing per frame. The principal axis needs the front resolved along
its own long axis and refuses when it is not. Skewness needs the head to burn
brighter than the back.

That last condition is worth stating plainly because it does not hold here. The
burning-cell source model gives every cell the same amplitude, so there is no
head-to-back intensity gradient for a third moment to read, and on this scene
the skewness estimator returns noise. It is kept, and kept reported, because
the condition is a property of the source model rather than of the method: a
scene whose cells radiate by mass loss rate would restore it. Where it cannot
work, the centroid offset reads the same downwind lean from the first moment
instead, which the geometry supplies whether or not the intensity does.

Agreement between drift and offset is therefore evidence rather than a
restatement, and the fusion reports its resultant length so that disagreement
shows up as a low number instead of being averaged away.

Angles are never averaged linearly. Every mean, spread and interval here is
circular, because the linear mean of 350 and 10 degrees is 180.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import theilslopes

from src.spark.inverse.centroid_bias_correction import (
    CentroidBiasField,
    apply_centroid_bias_correction,
)
from src.spark.inverse.map_moments import (
    compute_map_moments,
    compute_skewness_over_directions,
)
from src.spark.inverse.steered_response_power_sequence import (
    SteeredResponsePowerSequence,
)
from src.utils.array_types import Float64Array

CENTROID_DRIFT_METHOD: str = "centroid_drift"
SKEWNESS_METHOD: str = "skewness"
CENTROID_OFFSET_METHOD: str = "centroid_offset"
PRINCIPAL_AXIS_METHOD: str = "principal_axis"
FUSED_METHOD: str = "fused"

MINIMUM_DRIFT_FRAME_COUNT: int = 3
RESULTANT_FLOOR: float = 1e-12


@dataclass(frozen=True)
class BearingEstimate:
    """A direction with how tightly it is held.

    Attributes:
        bearing_rad: Direction of spread, in radians.
        circular_standard_deviation_rad: Spread of the estimate, circular.
        method: Which estimator produced it.
        frames_used: How many frames it was built from.
        resultant_length: Mean resultant length behind the spread, one when
            every resample agreed. Reported rather than only its logarithm
            because a fusion whose inputs disagree shows up here first.
    """

    bearing_rad: float
    circular_standard_deviation_rad: float
    method: str
    frames_used: int
    resultant_length: float


def compute_circular_mean_rad(angles_rad: Float64Array, weights: Float64Array) -> float:
    """Weighted mean direction of a set of angles.

    Args:
        angles_rad: Angles, shape `(n,)`.
        weights: Non-negative weights, same shape.

    Returns:
        The mean direction in radians, in `[-pi, pi]`.
    """
    angles = np.asarray(angles_rad, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    return float(
        np.arctan2(
            float(weight_values @ np.sin(angles)),
            float(weight_values @ np.cos(angles)),
        )
    )


def compute_resultant_length(angles_rad: Float64Array, weights: Float64Array) -> float:
    """How tightly a set of angles clusters, between zero and one.

    Args:
        angles_rad: Angles, shape `(n,)`.
        weights: Non-negative weights, same shape.

    Returns:
        The mean resultant length.
    """
    angles = np.asarray(angles_rad, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    total_weight = float(np.sum(weight_values))
    if total_weight <= 0.0:
        return 0.0
    return float(
        np.hypot(
            float(weight_values @ np.cos(angles)),
            float(weight_values @ np.sin(angles)),
        )
        / total_weight
    )


def compute_circular_standard_deviation_rad(resultant_length: float) -> float:
    """Circular spread implied by a resultant length.

    Args:
        resultant_length: Mean resultant length, in `[0, 1]`.

    Returns:
        The circular standard deviation in radians. A resultant of zero gives
        pi, the spread of a uniform direction, rather than infinity.
    """
    clamped = float(np.clip(resultant_length, RESULTANT_FLOOR, 1.0))
    return float(min(np.sqrt(-2.0 * np.log(clamped)), np.pi))


def compute_corrected_centroids_xy_m(
    sequence: SteeredResponsePowerSequence,
    background_percentile: float,
    bias_field: CentroidBiasField | None,
) -> Float64Array:
    """Centroid of every frame, with the array's own offset removed.

    Args:
        sequence: The map sequence.
        background_percentile: Percentile of a frame treated as background.
        bias_field: Tabulated offsets, or None to leave centroids uncorrected.

    Returns:
        Centroids of shape `(n_frames, 2)`.
    """
    maps = np.asarray(sequence.maps, dtype=np.float64)
    centroids_xy_m = np.empty((maps.shape[0], 2), dtype=np.float64)
    for frame_index in range(maps.shape[0]):
        centroid_xy_m = compute_map_moments(
            maps[frame_index],
            sequence.candidate_positions_xyz_m,
            background_percentile,
        ).centroid_xy_m
        centroids_xy_m[frame_index] = (
            centroid_xy_m
            if bias_field is None
            else apply_centroid_bias_correction(centroid_xy_m, bias_field)
        )
    return centroids_xy_m


def estimate_bearing_from_centroid_drift(
    sequence: SteeredResponsePowerSequence,
    background_percentile: float,
    bias_field: CentroidBiasField | None,
    minimum_displacement_m: float,
    bootstrap_resample_count: int,
    random_seed: int,
) -> BearingEstimate:
    """Fits the centroid's track against time and reads its direction.

    Theil-Sen rather than least squares, because the early frames of a run hold
    a handful of radiating cells and one bad centroid there would otherwise set
    the direction for the whole sequence.

    Args:
        sequence: The map sequence.
        background_percentile: Percentile of a frame treated as background.
        bias_field: Tabulated offsets, or None.
        minimum_displacement_m: Below this total drift the direction is noise.
        bootstrap_resample_count: Resamples over frames for the interval.
        random_seed: Seed making the bootstrap reproducible.

    Returns:
        The bearing with its circular spread.

    Raises:
        ValueError: If there are too few frames, or the centroid barely moved.
    """
    centroids_xy_m = compute_corrected_centroids_xy_m(
        sequence, background_percentile, bias_field
    )
    times_s = np.asarray(sequence.times_s, dtype=np.float64)
    if times_s.size < MINIMUM_DRIFT_FRAME_COUNT:
        raise ValueError(
            f"centroid drift needs at least {MINIMUM_DRIFT_FRAME_COUNT} frames, "
            f"got {times_s.size}"
        )
    displacement_m = float(np.linalg.norm(centroids_xy_m[-1] - centroids_xy_m[0]))
    if displacement_m < minimum_displacement_m:
        raise ValueError(
            f"the centroid moved {displacement_m:.2f} m, below the "
            f"{minimum_displacement_m:.2f} m needed for a direction to mean "
            "anything"
        )

    bearing_rad = _fit_drift_bearing_rad(times_s, centroids_xy_m)
    generator = np.random.default_rng(random_seed)
    resampled_rad = np.empty(bootstrap_resample_count, dtype=np.float64)
    for resample_index in range(bootstrap_resample_count):
        chosen = np.sort(generator.integers(0, times_s.size, times_s.size))
        if np.unique(times_s[chosen]).size < MINIMUM_DRIFT_FRAME_COUNT:
            resampled_rad[resample_index] = bearing_rad
            continue
        resampled_rad[resample_index] = _fit_drift_bearing_rad(
            times_s[chosen], centroids_xy_m[chosen]
        )
    weights = np.ones(bootstrap_resample_count, dtype=np.float64)
    resultant = compute_resultant_length(resampled_rad, weights)
    return BearingEstimate(
        bearing_rad=bearing_rad,
        circular_standard_deviation_rad=compute_circular_standard_deviation_rad(
            resultant
        ),
        method=CENTROID_DRIFT_METHOD,
        frames_used=int(times_s.size),
        resultant_length=resultant,
    )


def estimate_ignition_position_xy_m(
    sequence: SteeredResponsePowerSequence,
    background_percentile: float,
    bias_field: CentroidBiasField | None,
) -> Float64Array:
    """Where the fire started, from the earliest frame that carries mass.

    The first frames of a run hold a handful of radiating cells within a metre
    or two of the ignition point, so their centroid is the ignition point to
    within the grid. Every later estimate that needs an origin uses this rather
    than being told the truth.

    Args:
        sequence: The map sequence.
        background_percentile: Percentile of a frame treated as background.
        bias_field: Tabulated offsets, or None.

    Returns:
        The ignition position, shape `(2,)`.

    Raises:
        ValueError: If the sequence holds no frame.
    """
    maps = np.asarray(sequence.maps, dtype=np.float64)
    if maps.shape[0] == 0:
        raise ValueError("an empty sequence has no ignition point")
    centroids_xy_m = compute_corrected_centroids_xy_m(
        sequence, background_percentile, bias_field
    )
    return np.asarray(centroids_xy_m[0], dtype=np.float64)


def estimate_bearing_from_centroid_offset(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    ignition_position_xy_m: Float64Array,
    background_percentile: float,
    minimum_offset_m: float,
) -> BearingEstimate:
    """Reads the direction from the ignition point to the frame's centroid.

    A front that has burnt out behind itself sits entirely downwind of where it
    started, so its centroid leans that way whether or not its head burns
    brighter than its back. This is a first moment where the skewness estimator
    is a third, which makes it far less sensitive to the map's tails and lets it
    work on a source model with no intensity gradient at all.

    Args:
        map_values: One frame, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        ignition_position_xy_m: Where the fire started, shape `(2,)`.
        background_percentile: Percentile of the frame treated as background.
        minimum_offset_m: Below this the centroid has not left the ignition
            point and its direction is noise.

    Returns:
        The bearing, with a spread that tightens as the offset grows.

    Raises:
        ValueError: If the centroid has not moved far enough to point anywhere.
    """
    moments = compute_map_moments(
        map_values, candidate_positions_xyz_m, background_percentile
    )
    offset_xy_m = moments.centroid_xy_m - np.asarray(
        ignition_position_xy_m, dtype=np.float64
    )
    offset_m = float(np.linalg.norm(offset_xy_m))
    if offset_m < minimum_offset_m:
        raise ValueError(
            f"the centroid sits {offset_m:.2f} m from the ignition point, below "
            f"the {minimum_offset_m:.2f} m needed for a direction to mean anything"
        )
    resultant = float(offset_m / (offset_m + minimum_offset_m))
    return BearingEstimate(
        bearing_rad=float(np.arctan2(offset_xy_m[1], offset_xy_m[0])),
        circular_standard_deviation_rad=compute_circular_standard_deviation_rad(
            resultant
        ),
        method=CENTROID_OFFSET_METHOD,
        frames_used=1,
        resultant_length=resultant,
    )


def estimate_bearing_from_skewness(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    background_percentile: float,
    direction_count: int,
) -> BearingEstimate:
    """Reads the direction the map's mass leans, from one frame alone.

    A head fire burns brighter than its back, so the mass is lopsided downwind
    even when the front itself is not resolved. This needs no motion, which is
    what makes it independent of the drift estimator rather than a check on it.

    **The bearing is the direction of least skewness, not most.** The plan
    prescribes the argmax and that is the wrong sign. A third moment measures
    which way a distribution's long tail points, and concentrating mass toward
    the head leaves the tail behind it: on a ring whose downwind half is twice
    as bright, the skewness along the true bearing is -0.51 and along its
    reverse is +0.51. Taking the argmax returns the bearing rotated by half a
    turn, which on a scene with no other estimator to contradict it would pass
    unnoticed.

    Args:
        map_values: One frame, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        background_percentile: Percentile of the frame treated as background.
        direction_count: Directions swept over a full turn.

    Returns:
        The bearing, with a spread taken from how far the sweep's peak stands
        above its own range. A sweep with no lean at all has nothing to report
        and returns a resultant near a half.
    """
    moments = compute_map_moments(
        map_values, candidate_positions_xyz_m, background_percentile
    )
    angles_rad, skewness = compute_skewness_over_directions(
        map_values,
        candidate_positions_xyz_m,
        moments.centroid_xy_m,
        direction_count,
        background_percentile,
    )
    leaning = -skewness
    peak_index = int(np.argmax(leaning))
    bearing_rad = _refine_angular_peak_rad(angles_rad, leaning, peak_index)
    span = float(np.max(leaning) - np.min(leaning))
    resultant = float(
        np.clip(float(np.max(leaning)) / span, 0.0, 1.0) if span > 0.0 else 0.0
    )
    return BearingEstimate(
        bearing_rad=bearing_rad,
        circular_standard_deviation_rad=compute_circular_standard_deviation_rad(
            resultant
        ),
        method=SKEWNESS_METHOD,
        frames_used=1,
        resultant_length=resultant,
    )


def estimate_bearing_from_principal_axis(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    ignition_position_xy_m: Float64Array,
    background_percentile: float,
    point_spread_covariance_m2: Float64Array,
) -> BearingEstimate:
    """Takes the long axis of what is left after the response is subtracted.

    The axis is only defined modulo half a turn, so it is disambiguated by
    which way the centroid has moved from the ignition point.

    Args:
        map_values: One frame, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        ignition_position_xy_m: Where the fire started, shape `(2,)`.
        background_percentile: Percentile of the frame treated as background.
        point_spread_covariance_m2: The array response's own covariance.

    Returns:
        The bearing.

    Raises:
        ValueError: If the front is not resolved along its own long axis, which
            leaves the subtraction with a non-positive eigenvalue and the axis
            meaningless rather than merely imprecise.
    """
    moments = compute_map_moments(
        map_values, candidate_positions_xyz_m, background_percentile
    )
    source_covariance_m2 = moments.covariance_xy_m2 - np.asarray(
        point_spread_covariance_m2, dtype=np.float64
    )
    eigenvalues, eigenvectors = np.linalg.eigh(source_covariance_m2)
    if float(np.max(eigenvalues)) <= 0.0:
        raise ValueError(
            "the front is not resolved along any axis once the array response "
            "is subtracted, so its orientation is undefined"
        )
    major_direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    spread_vector = moments.centroid_xy_m - np.asarray(
        ignition_position_xy_m, dtype=np.float64
    )
    if float(spread_vector @ major_direction) < 0.0:
        major_direction = -major_direction
    eigenvalue_ratio = float(
        np.min(np.clip(eigenvalues, 0.0, None)) / np.max(eigenvalues)
    )
    return BearingEstimate(
        bearing_rad=float(np.arctan2(major_direction[1], major_direction[0])),
        circular_standard_deviation_rad=compute_circular_standard_deviation_rad(
            1.0 - eigenvalue_ratio
        ),
        method=PRINCIPAL_AXIS_METHOD,
        frames_used=1,
        resultant_length=1.0 - eigenvalue_ratio,
    )


def fuse_bearing_estimates(
    estimates: list[BearingEstimate], weights: dict[str, float]
) -> BearingEstimate:
    """Combines directions as unit vectors, weighted by method and confidence.

    Args:
        estimates: What each method returned.
        weights: Weight per method name.

    Returns:
        The fused bearing. Its resultant length is the consistency diagnostic:
        a low value with individually confident inputs means the methods
        disagree, which is a finding rather than something to average away.

    Raises:
        ValueError: If there is nothing to fuse.
    """
    if not estimates:
        raise ValueError("there is no bearing to fuse")
    angles_rad = np.array([estimate.bearing_rad for estimate in estimates])
    confidence = np.array(
        [
            weights.get(estimate.method, 1.0)
            / max(estimate.circular_standard_deviation_rad**2, 1e-6)
            for estimate in estimates
        ]
    )
    resultant = compute_resultant_length(angles_rad, confidence)
    return BearingEstimate(
        bearing_rad=compute_circular_mean_rad(angles_rad, confidence),
        circular_standard_deviation_rad=compute_circular_standard_deviation_rad(
            resultant
        ),
        method=FUSED_METHOD,
        frames_used=sum(estimate.frames_used for estimate in estimates),
        resultant_length=resultant,
    )


def _fit_drift_bearing_rad(
    times_s: Float64Array, centroids_xy_m: Float64Array
) -> float:
    """Theil-Sen slope of each coordinate against time, as a direction.

    Args:
        times_s: Frame times, shape `(n_frames,)`.
        centroids_xy_m: Frame centroids, shape `(n_frames, 2)`.

    Returns:
        The bearing in radians.
    """
    slope_x = float(theilslopes(centroids_xy_m[:, 0], times_s)[0])
    slope_y = float(theilslopes(centroids_xy_m[:, 1], times_s)[0])
    return float(np.arctan2(slope_y, slope_x))


def _refine_angular_peak_rad(
    angles_rad: Float64Array, values: Float64Array, peak_index: int
) -> float:
    """Parabolic refinement of a swept maximum, wrapping at the ends.

    Args:
        angles_rad: Swept directions, ascending and evenly spaced.
        values: Value at each direction.
        peak_index: Index of the largest value.

    Returns:
        The refined direction in radians.
    """
    angles = np.asarray(angles_rad, dtype=np.float64)
    sampled = np.asarray(values, dtype=np.float64)
    step_rad = float(angles[1] - angles[0])
    before = sampled[(peak_index - 1) % sampled.size]
    at = sampled[peak_index]
    after = sampled[(peak_index + 1) % sampled.size]
    denominator = before - 2.0 * at + after
    offset = 0.0 if abs(denominator) <= 0.0 else 0.5 * (before - after) / denominator
    return float(angles[peak_index] + float(np.clip(offset, -1.0, 1.0)) * step_rad)
