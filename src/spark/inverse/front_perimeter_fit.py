"""A front described by a handful of numbers instead of a whole density.

The robust middle ground between reading a distance off one profile and solving
for a free-form source density: enough parameters to carry a curved front, few
enough that thousands of map cells overdetermine them.

**The perimeter is an elliptical arc, not a closed ellipse.** The plan fits a
closed curve carrying angularly varying intensity. On a fire that burns out
behind itself the radiating front covers about ninety degrees, so a closed
model would put perimeter across two hundred and seventy degrees of silence and
the fit would trade real front against invented front to balance it. Carrying
the angular support as two fitted parameters instead lets the model report how
much of the circle radiates, which is a measurement worth having rather than an
assumption worth avoiding.

The model is rendered and blurred through the same response operator the
deconvolution uses, so the parametric and free-form routes are scored against
the map on identical terms.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.spark.inverse.map_deconvolution import PointSpreadOperator, apply_operator
from src.utils.array_types import Float64Array

DENSITY_FLOOR: float = 1e-12
PARAMETER_COUNT: int = 7
# Finite-difference step for the Jacobian, as a fraction of each parameter. The
# default is near machine epsilon, which on a rendered density moves nothing.
DIFFERENCE_STEP: float = 1e-3


@dataclass(frozen=True)
class PerimeterParameters:
    """An elliptical arc, as the numbers that place it.

    Attributes:
        centre_xy_m: Centre of the ellipse, shape `(2,)`.
        semi_axis_major_m: Major semi-axis, in metres.
        semi_axis_minor_m: Minor semi-axis, in metres.
        orientation_rad: Direction of the major axis.
        sector_centre_rad: Bearing, from the origin, of the middle of the
            radiating arc.
        sector_half_width_rad: Half the angular extent that radiates. Half a
            turn means the whole ellipse radiates.
    """

    centre_xy_m: Float64Array
    semi_axis_major_m: float
    semi_axis_minor_m: float
    orientation_rad: float
    sector_centre_rad: float
    sector_half_width_rad: float


@dataclass(frozen=True)
class PerimeterFit:
    """What the fit settled on and how well it explained the map.

    Attributes:
        parameters: The fitted arc.
        correlation: Correlation between the blurred model and the map.
        residual: Root-mean-square residual after scaling to unit mass.
        converged: Whether the optimiser reported success.
    """

    parameters: PerimeterParameters
    correlation: float
    residual: float
    converged: bool


def render_perimeter_density(
    parameters: PerimeterParameters,
    candidate_positions_xyz_m: Float64Array,
    origin_xy_m: Float64Array,
    grid_shape: tuple[int, int],
    sample_count: int,
) -> Float64Array:
    """Lays the arc's mass onto the candidate grid, spread across cells.

    Deposition is bilinear rather than to the nearest cell, and that is a
    correctness requirement rather than a refinement. Rounding each sample to
    one cell makes the rendered density — and so the fit's whole objective —
    piecewise constant in the parameters: a small change moves no cell, the
    finite-difference Jacobian comes back exactly zero, and the optimiser
    reports success without having moved off its seed.

    Args:
        parameters: The arc to render.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        origin_xy_m: Point the angular sector is measured from.
        grid_shape: `(n_y, n_x)` of the grid.
        sample_count: Points sampled along the arc.

    Returns:
        A density of shape `(n_y, n_x)`, summing to one when the arc lands on
        the grid at all.
    """
    positions = np.atleast_2d(np.asarray(candidate_positions_xyz_m, dtype=np.float64))
    x_m = np.unique(positions[:, 0])
    y_m = np.unique(positions[:, 1])
    spacing_m = float(x_m[1] - x_m[0]) if x_m.size > 1 else 1.0

    angles_rad = np.linspace(
        parameters.sector_centre_rad - parameters.sector_half_width_rad,
        parameters.sector_centre_rad + parameters.sector_half_width_rad,
        sample_count,
    )
    radii_m = compute_front_distance_by_angle(parameters, origin_xy_m, angles_rad)
    finite = np.isfinite(radii_m)
    density = np.zeros(grid_shape, dtype=np.float64)
    if not bool(np.any(finite)):
        return density

    origin = np.asarray(origin_xy_m, dtype=np.float64)
    points_xy_m = origin + np.stack(
        (
            radii_m[finite] * np.cos(angles_rad[finite]),
            radii_m[finite] * np.sin(angles_rad[finite]),
        ),
        axis=1,
    )
    column_real = (points_xy_m[:, 0] - x_m[0]) / spacing_m
    row_real = (points_xy_m[:, 1] - y_m[0]) / spacing_m
    column = np.clip(np.floor(column_real).astype(np.int64), 0, x_m.size - 2)
    row = np.clip(np.floor(row_real).astype(np.int64), 0, y_m.size - 2)
    column_weight = np.clip(column_real - column, 0.0, 1.0)
    row_weight = np.clip(row_real - row, 0.0, 1.0)

    np.add.at(density, (row, column), (1.0 - row_weight) * (1.0 - column_weight))
    np.add.at(density, (row, column + 1), (1.0 - row_weight) * column_weight)
    np.add.at(density, (row + 1, column), row_weight * (1.0 - column_weight))
    np.add.at(density, (row + 1, column + 1), row_weight * column_weight)
    total = float(np.sum(density))
    return density / total if total > 0.0 else density


def compute_front_distance_by_angle(
    parameters: PerimeterParameters,
    origin_xy_m: Float64Array,
    angles_rad: Float64Array,
) -> Float64Array:
    """Where a ray from the origin meets the arc, per direction.

    Args:
        parameters: The fitted arc.
        origin_xy_m: Where rays start, shape `(2,)`.
        angles_rad: Directions to evaluate, shape `(n_angles,)`.

    Returns:
        Distance at each angle, shape `(n_angles,)`, `nan` outside the
        radiating sector or where the ray misses the ellipse entirely.
    """
    angles = np.asarray(angles_rad, dtype=np.float64)
    offset_rad = np.arctan2(
        np.sin(angles - parameters.sector_centre_rad),
        np.cos(angles - parameters.sector_centre_rad),
    )
    inside_sector = np.abs(offset_rad) <= parameters.sector_half_width_rad

    cosine = np.cos(parameters.orientation_rad)
    sine = np.sin(parameters.orientation_rad)
    to_origin = np.asarray(origin_xy_m, dtype=np.float64) - np.asarray(
        parameters.centre_xy_m, dtype=np.float64
    )
    origin_along = cosine * to_origin[0] + sine * to_origin[1]
    origin_across = -sine * to_origin[0] + cosine * to_origin[1]

    direction_x = np.cos(angles)
    direction_y = np.sin(angles)
    along = cosine * direction_x + sine * direction_y
    across = -sine * direction_x + cosine * direction_y

    major = max(parameters.semi_axis_major_m, DENSITY_FLOOR)
    minor = max(parameters.semi_axis_minor_m, DENSITY_FLOOR)
    quadratic = (along / major) ** 2 + (across / minor) ** 2
    linear = 2.0 * (origin_along * along / major**2 + origin_across * across / minor**2)
    constant = (origin_along / major) ** 2 + (origin_across / minor) ** 2 - 1.0

    discriminant = linear**2 - 4.0 * quadratic * constant
    has_root = (discriminant >= 0.0) & (quadratic > 0.0)
    root = np.sqrt(np.where(has_root, discriminant, 0.0))
    outer = np.where(
        has_root,
        (-linear + root) / (2.0 * np.maximum(quadratic, DENSITY_FLOOR)),
        np.nan,
    )
    return np.asarray(
        np.where(inside_sector & has_root & (outer > 0.0), outer, np.nan),
        dtype=np.float64,
    )


def fit_front_perimeter(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    operator: PointSpreadOperator,
    origin_xy_m: Float64Array,
    initial_parameters: PerimeterParameters,
    maximum_iterations: int,
    sample_count: int,
) -> PerimeterFit:
    """Fits an elliptical arc to one prepared map.

    Args:
        map_values: The prepared map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        operator: The response, used to blur the model before scoring.
        origin_xy_m: Point the angular sector is measured from.
        initial_parameters: Seed, normally from the bearing and extent stages.
        maximum_iterations: Iteration ceiling for the optimiser.
        sample_count: Points sampled along the arc when rendering.

    Returns:
        The fit, with the correlation it achieved.
    """
    observed = np.asarray(map_values, dtype=np.float64).reshape(operator.grid_shape)
    observed_unit = observed / max(float(np.sum(observed)), DENSITY_FLOOR)

    def unpack(vector: Float64Array) -> PerimeterParameters:
        return PerimeterParameters(
            centre_xy_m=np.array([vector[0], vector[1]]),
            semi_axis_major_m=float(vector[2]),
            semi_axis_minor_m=float(vector[3]),
            orientation_rad=float(vector[4]),
            sector_centre_rad=float(vector[5]),
            sector_half_width_rad=float(vector[6]),
        )

    def residuals(vector: Float64Array) -> Float64Array:
        density = render_perimeter_density(
            unpack(vector),
            candidate_positions_xyz_m,
            origin_xy_m,
            operator.grid_shape,
            sample_count,
        )
        blurred = apply_operator(operator, density)
        blurred_unit = blurred / max(float(np.sum(blurred)), DENSITY_FLOOR)
        return np.asarray((blurred_unit - observed_unit).ravel(), dtype=np.float64)

    seed = np.array(
        [
            float(initial_parameters.centre_xy_m[0]),
            float(initial_parameters.centre_xy_m[1]),
            initial_parameters.semi_axis_major_m,
            initial_parameters.semi_axis_minor_m,
            initial_parameters.orientation_rad,
            initial_parameters.sector_centre_rad,
            initial_parameters.sector_half_width_rad,
        ]
    )
    lower = seed - np.array([15.0, 15.0, 12.0, 12.0, np.pi, np.pi, 0.0])
    upper = seed + np.array([15.0, 15.0, 12.0, 12.0, np.pi, np.pi, 0.0])
    lower[2] = 0.5
    lower[3] = 0.3
    upper[2] = 30.0
    upper[3] = 30.0
    lower[6] = 0.15
    upper[6] = np.pi

    solution = least_squares(
        residuals,
        x0=seed,
        bounds=(lower, upper),
        max_nfev=maximum_iterations,
        diff_step=DIFFERENCE_STEP,
    )
    fitted = unpack(solution.x)
    blurred = apply_operator(
        operator,
        render_perimeter_density(
            fitted,
            candidate_positions_xyz_m,
            origin_xy_m,
            operator.grid_shape,
            sample_count,
        ),
    )
    return PerimeterFit(
        parameters=fitted,
        correlation=float(np.corrcoef(blurred.ravel(), observed.ravel())[0, 1]),
        residual=float(np.sqrt(np.mean(solution.fun**2))),
        converged=bool(solution.success),
    )
