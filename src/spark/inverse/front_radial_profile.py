"""Where the front sits along one line through the map.

The cheapest thing that yields a distance, and the first place to look before
any deconvolution. A profile taken along the bearing through the centroid is
one dimension instead of two, so it can be matched against a model with a
handful of parameters rather than solved for a whole density.

Reading the distance off the profile's maximum would not work: the response is
a metre and a half wide and the front is only a few metres across, so the peak
of a smeared profile sits wherever two lobes happen to overlap. Fitting a model
that carries the response explicitly uses the whole curve instead of one point.

**A single lobe finds the band's centre, not its leading edge, so there is a
band model too.** Measured on synthetic bands, the shortfall between a
one-lobe fit and the outer edge is exactly the band's own half-width and is
completely independent of the response: a band of half-width 3.5 m under-reads
by 3.53 m whether the response is 0.8 m or 2.5 m across. That is the one-lobe
fit working correctly and being asked the wrong question. `estimate_front_band`
fits the two edges directly, which is what a front position actually wants.

**The lobe model keeps a back lobe whose amplitude is fitted rather than
assumed.**
The plan prescribes a two-lobe model — head ahead of the origin, back behind
it — on the picture of a front that radiates all the way round. On a fire that
burns out behind itself there is no back lobe at all, and forcing one makes the
head distance absorb the error. Fitting the ratio instead means the model
reports which case it is in: a ratio near zero says the front is one-sided, and
that is a measurement rather than a modelling choice.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.utils.array_types import Float64Array

ONE_SIDED_RATIO_THRESHOLD: float = 0.15
PROFILE_FLOOR: float = 1e-12
# Edge softness of the band model, as a multiple of the profile step. A hard
# box is piecewise constant in its own edges at the sampling resolution, which
# leaves the optimiser a Jacobian of exactly zero and stops it on its seed.
BAND_EDGE_SOFTNESS: float = 1.5


@dataclass(frozen=True)
class FrontDistanceEstimate:
    """What one profile says about where the front is.

    Attributes:
        head_distance_m: Distance from the origin to the leading lobe.
        back_distance_m: Distance to the trailing lobe, meaningful only when
            the ratio says there is one.
        back_to_head_ratio: Amplitude of the trailing lobe over the leading
            one. Near zero on a front that has burnt out behind itself.
        residual: Root-mean-square fit residual over the profile's own peak.
        is_one_sided: Whether the trailing lobe is too faint to be a lobe,
            which is the plan's two-lobe model reporting its own inadequacy.
    """

    head_distance_m: float
    back_distance_m: float
    back_to_head_ratio: float
    residual: float
    is_one_sided: bool


def build_grid_axes(
    candidate_positions_xyz_m: Float64Array,
) -> tuple[Float64Array, Float64Array]:
    """Recovers the two axes of a regular candidate grid.

    Args:
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.

    Returns:
        `(x_m, y_m)`, both ascending and unique.
    """
    positions = np.atleast_2d(np.asarray(candidate_positions_xyz_m, dtype=np.float64))
    return np.unique(positions[:, 0]), np.unique(positions[:, 1])


def sample_map_bilinear(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    query_positions_xy_m: Float64Array,
) -> Float64Array:
    """Reads a map at arbitrary positions by bilinear interpolation.

    Args:
        map_values: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        query_positions_xy_m: Where to read, shape `(n_queries, 2)`.

    Returns:
        Map value at each query, shape `(n_queries,)`. Queries outside the grid
        are clamped to its edge rather than extrapolated.
    """
    x_m, y_m = build_grid_axes(candidate_positions_xyz_m)
    grid = np.asarray(map_values, dtype=np.float64).reshape(y_m.size, x_m.size)
    queries = np.atleast_2d(np.asarray(query_positions_xy_m, dtype=np.float64))

    column = np.clip(np.searchsorted(x_m, queries[:, 0]) - 1, 0, max(x_m.size - 2, 0))
    row = np.clip(np.searchsorted(y_m, queries[:, 1]) - 1, 0, max(y_m.size - 2, 0))
    next_column = np.minimum(column + 1, x_m.size - 1)
    next_row = np.minimum(row + 1, y_m.size - 1)

    column_span = np.where(
        x_m[next_column] > x_m[column], x_m[next_column] - x_m[column], 1.0
    )
    row_span = np.where(y_m[next_row] > y_m[row], y_m[next_row] - y_m[row], 1.0)
    column_weight = np.clip((queries[:, 0] - x_m[column]) / column_span, 0.0, 1.0)
    row_weight = np.clip((queries[:, 1] - y_m[row]) / row_span, 0.0, 1.0)

    lower = (1.0 - column_weight) * grid[row, column] + column_weight * grid[
        row, next_column
    ]
    upper = (1.0 - column_weight) * grid[next_row, column] + column_weight * grid[
        next_row, next_column
    ]
    return np.asarray((1.0 - row_weight) * lower + row_weight * upper, dtype=np.float64)


def extract_radial_profile(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    origin_xy_m: Float64Array,
    direction_xy: Float64Array,
    maximum_radius_m: float,
    radius_step_m: float,
) -> tuple[Float64Array, Float64Array]:
    """Samples the map along a line through an origin, both ways.

    Args:
        map_values: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        origin_xy_m: Where the line passes through, shape `(2,)`.
        direction_xy: Direction of positive radius, shape `(2,)`.
        maximum_radius_m: Furthest radius sampled, in each direction.
        radius_step_m: Spacing along the line, in metres.

    Returns:
        `(radii_m, profile_values)`, both shape `(n_samples,)`, with radii
        ascending from `-maximum_radius_m` to `+maximum_radius_m`.

    Raises:
        ValueError: If the step or the radius is not positive, or the direction
            has no length.
    """
    if radius_step_m <= 0.0 or maximum_radius_m <= 0.0:
        raise ValueError("the profile step and maximum radius must both be positive")
    direction = np.asarray(direction_xy, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        raise ValueError("a profile direction must have non-zero length")

    radii_m = np.arange(
        -maximum_radius_m, maximum_radius_m + radius_step_m, radius_step_m
    )
    positions_xy_m = np.asarray(origin_xy_m, dtype=np.float64) + np.outer(
        radii_m, direction / length
    )
    return radii_m, sample_map_bilinear(
        map_values, candidate_positions_xyz_m, positions_xy_m
    )


def build_two_lobe_model(
    radii_m: Float64Array,
    response_radii_m: Float64Array,
    response_profile: Float64Array,
    head_distance_m: float,
    back_distance_m: float,
    back_to_head_ratio: float,
) -> Float64Array:
    """A leading lobe and a trailing one, each carrying the array response.

    Args:
        radii_m: Radii the model is evaluated at, shape `(n_samples,)`.
        response_radii_m: Radii of the response's own profile.
        response_profile: That profile, peaking at zero radius.
        head_distance_m: Where the leading lobe sits.
        back_distance_m: Where the trailing lobe sits, as a positive distance.
        back_to_head_ratio: Amplitude of the trailing lobe over the leading one.

    Returns:
        The model at each radius, shape `(n_samples,)`, peaking at one.
    """
    head = np.interp(
        np.asarray(radii_m) - head_distance_m,
        response_radii_m,
        response_profile,
        left=0.0,
        right=0.0,
    )
    back = np.interp(
        np.asarray(radii_m) + back_distance_m,
        response_radii_m,
        response_profile,
        left=0.0,
        right=0.0,
    )
    return np.asarray(head + back_to_head_ratio * back, dtype=np.float64)


@dataclass(frozen=True)
class FrontBandEstimate:
    """The radiating band's two edges along one profile line.

    Attributes:
        inner_edge_m: Where the band starts, as a distance from the origin.
        outer_edge_m: Where it ends — the front's leading edge.
        half_width_m: Half the band's radial extent.
        residual: Root-mean-square fit residual over the profile's own peak.
    """

    inner_edge_m: float
    outer_edge_m: float
    half_width_m: float
    residual: float


def build_band_model(
    radii_m: Float64Array,
    response_radii_m: Float64Array,
    response_profile: Float64Array,
    inner_edge_m: float,
    outer_edge_m: float,
) -> Float64Array:
    """A radiating band between two edges, blurred by the array response.

    The band's edges are soft rather than square. A hard box changes only when
    an edge crosses a sample, so its derivative with respect to either edge is
    zero almost everywhere and the optimiser has nothing to follow.

    Args:
        radii_m: Radii the model is evaluated at, uniformly spaced.
        response_radii_m: Radii of the response's own profile.
        response_profile: That profile, peaking at zero radius.
        inner_edge_m: Where the band starts.
        outer_edge_m: Where it ends.

    Returns:
        The model at each radius, shape `(n_samples,)`, peaking at one.
    """
    radii = np.asarray(radii_m, dtype=np.float64)
    step_m = float(radii[1] - radii[0]) if radii.size > 1 else 1.0
    softness_m = max(BAND_EDGE_SOFTNESS * step_m, PROFILE_FLOOR)
    band = _soft_step(radii - inner_edge_m, softness_m) * _soft_step(
        outer_edge_m - radii, softness_m
    )

    kernel = np.interp(
        radii - float(np.mean(radii)),
        response_radii_m,
        response_profile,
        left=0.0,
        right=0.0,
    )
    total = float(np.sum(kernel))
    if total <= PROFILE_FLOOR:
        return np.asarray(band, dtype=np.float64)
    blurred = np.convolve(band, kernel / total, mode="same")
    peak = float(np.max(blurred))
    return np.asarray(blurred / peak if peak > 0.0 else blurred, dtype=np.float64)


def estimate_front_band_by_matched_filter(
    radii_m: Float64Array,
    profile_values: Float64Array,
    response_radii_m: Float64Array,
    response_profile: Float64Array,
    maximum_distance_m: float,
) -> FrontBandEstimate:
    """Fits both edges of the radiating band to one profile.

    Args:
        radii_m: Radii of the measured profile, uniformly spaced.
        profile_values: The measured profile, same shape.
        response_radii_m: Radii of the response's own profile.
        response_profile: That profile, peaking at zero radius.
        maximum_distance_m: Largest radius an edge may be placed at.

    Returns:
        The fitted band, whose outer edge is the front's leading edge.

    Raises:
        ValueError: If the profile carries no signal to fit.
    """
    radii = np.asarray(radii_m, dtype=np.float64)
    measured = np.asarray(profile_values, dtype=np.float64)
    peak = float(np.max(measured))
    if peak <= PROFILE_FLOOR:
        raise ValueError("the profile carries no signal, so no band can be fitted")
    normalised = measured / peak

    def residuals(parameters: Float64Array) -> Float64Array:
        inner_edge_m, outer_edge_m, scale = parameters
        model = scale * build_band_model(
            radii,
            response_radii_m,
            response_profile,
            float(inner_edge_m),
            float(max(outer_edge_m, inner_edge_m + 1e-3)),
        )
        return np.asarray(model - normalised, dtype=np.float64)

    seed_centre_m = _locate_forward_peak_m(radii, normalised)
    solution = least_squares(
        residuals,
        x0=np.array([max(seed_centre_m - 1.0, 0.0), seed_centre_m + 1.0, 1.0]),
        bounds=(
            np.array([-maximum_distance_m, 0.0, 0.1]),
            np.array([maximum_distance_m, maximum_distance_m, 10.0]),
        ),
        diff_step=1e-3,
    )
    inner_edge_m, outer_edge_m, _ = solution.x
    return FrontBandEstimate(
        inner_edge_m=float(min(inner_edge_m, outer_edge_m)),
        outer_edge_m=float(max(inner_edge_m, outer_edge_m)),
        half_width_m=float(abs(outer_edge_m - inner_edge_m) / 2.0),
        residual=float(np.sqrt(np.mean(solution.fun**2))),
    )


def _locate_forward_peak_m(
    radii_m: Float64Array, profile_values: Float64Array
) -> float:
    """Radius of the profile's brightest point ahead of the origin.

    A head distance is the front's distance *in the direction it was asked
    about*, so the search runs over non-negative radii only. On a fire that
    burns out behind itself the distinction is idle — there is nothing behind
    to find — but a fire that spreads upwind as well puts a lobe on each side,
    and seeding from the profile's global maximum lets the fit converge on the
    one behind the origin and report a head distance of, say, minus four
    metres. Measured on `configs/f2`, that happened on four frames of ten.

    Args:
        radii_m: Radii of the measured profile, ascending.
        profile_values: The measured profile, same shape.

    Returns:
        The radius of the largest forward sample, or zero when the profile
        reaches no further than the origin.
    """
    radii = np.asarray(radii_m, dtype=np.float64)
    forward = radii >= 0.0
    if not bool(np.any(forward)):
        return 0.0
    forward_values = np.where(forward, np.asarray(profile_values), -np.inf)
    return float(radii[int(np.argmax(forward_values))])


def _soft_step(offsets_m: Float64Array, softness_m: float) -> Float64Array:
    """A step that rises over a finite width, so its edge has a derivative.

    Args:
        offsets_m: Distance past the edge, negative before it.
        softness_m: Width the step rises over.

    Returns:
        Values in `(0, 1)`, same shape.
    """
    return np.asarray(
        0.5 * (1.0 + np.tanh(np.asarray(offsets_m, dtype=np.float64) / softness_m)),
        dtype=np.float64,
    )


def estimate_front_distance_by_matched_filter(
    radii_m: Float64Array,
    profile_values: Float64Array,
    response_radii_m: Float64Array,
    response_profile: Float64Array,
    maximum_distance_m: float,
) -> FrontDistanceEstimate:
    """Fits the two-lobe model to one profile and reports what it found.

    Args:
        radii_m: Radii of the measured profile, shape `(n_samples,)`.
        profile_values: The measured profile, same shape.
        response_radii_m: Radii of the response's own profile.
        response_profile: That profile, peaking at zero radius.
        maximum_distance_m: Largest distance either lobe may be placed at.

    Returns:
        The fitted distances with the residual and the one-sided flag.

    Raises:
        ValueError: If the profile carries no signal to fit.
    """
    radii = np.asarray(radii_m, dtype=np.float64)
    measured = np.asarray(profile_values, dtype=np.float64)
    peak = float(np.max(measured))
    if peak <= PROFILE_FLOOR:
        raise ValueError("the profile carries no signal, so no distance can be fitted")
    normalised = measured / peak

    def residuals(parameters: Float64Array) -> Float64Array:
        head_distance_m, back_distance_m, ratio, scale = parameters
        model = scale * build_two_lobe_model(
            radii,
            response_radii_m,
            response_profile,
            float(head_distance_m),
            float(back_distance_m),
            float(ratio),
        )
        return np.asarray(model - normalised, dtype=np.float64)

    seed_distance_m = float(radii[int(np.argmax(normalised))])
    solution = least_squares(
        residuals,
        x0=np.array([max(seed_distance_m, 0.5), 0.5, 0.3, 1.0]),
        bounds=(
            np.array([0.0, 0.0, 0.0, 0.1]),
            np.array([maximum_distance_m, maximum_distance_m, 2.0, 10.0]),
        ),
    )
    head_distance_m, back_distance_m, ratio, _ = solution.x
    return FrontDistanceEstimate(
        head_distance_m=float(head_distance_m),
        back_distance_m=float(back_distance_m),
        back_to_head_ratio=float(ratio),
        residual=float(np.sqrt(np.mean(solution.fun**2))),
        is_one_sided=bool(ratio < ONE_SIDED_RATIO_THRESHOLD),
    )
