"""Solving the imaging map back to a source density.

The map is the source density convolved with the array response, so recovering
the density is a deconvolution — the same family the deflation loop belongs to,
with the sparse point-source assumption dropped. That assumption is what caps
the deflation loop on a distributed front, so dropping it is the point.

**The operator is applied by transform, not built.** The plan asks for a
`LinearOperator` applying the response on the fly, because the dense matrix is
`n_grid` by `n_grid` — 14400 squared, about 1.6 GB. An on-the-fly matvec is
still `n_grid` squared arithmetic per iteration. Treating the response as
shift-invariant instead makes every application one pair of transforms, which
is what puts Richardson-Lucy within reach at all: a shift-variant operator
would need a response evaluated at every cell, and one response costs eight
seconds.

**That shift-invariance is measured, not assumed.** Re-centring the response
computed at seven positions across the fire's own region and correlating each
against the centred one gives 0.935 to 0.966. It is an approximation and the
figures above are its fidelity; a front that wandered towards the receiver ring
would need the operator rebuilt about its new centre.

**Total variation is physically motivated here rather than generic.** Only the
perimeter of a fire radiates and the burnt interior is silent, so the true
support is one-dimensional and a penalty that prefers piecewise-constant
regions with sharp edges is preferring the right thing.

Convolution is circular. The domain is sixty metres and the response about
three, so wrap-around touches only cells at the boundary, which no front in
this scene reaches.
"""

from dataclasses import dataclass

import numpy as np

from src.utils.array_types import ComplexArray, Float64Array

DENSITY_FLOOR: float = 1e-12
GRADIENT_FLOOR: float = 1e-9


@dataclass(frozen=True)
class PointSpreadOperator:
    """The array response as something that can be applied to a density.

    Attributes:
        kernel_spectrum: Transform of the response, centred at the origin.
        grid_shape: `(n_y, n_x)` of the maps it acts on.
        grid_spacing_m: Cell side length, in metres.
        normalisation: Sum of the response, so the operator conserves mass.
    """

    kernel_spectrum: ComplexArray
    grid_shape: tuple[int, int]
    grid_spacing_m: float
    normalisation: float


def build_point_spread_operator(
    response_map: Float64Array, grid_shape: tuple[int, int], grid_spacing_m: float
) -> PointSpreadOperator:
    """Prepares a response for repeated application.

    Args:
        response_map: The response to a source at the grid centre, shape
            `(n_cells,)`.
        grid_shape: `(n_y, n_x)` of the grid it was computed on.
        grid_spacing_m: Cell side length, in metres.

    Returns:
        The operator.

    Raises:
        ValueError: If the response does not fill the stated grid, or carries
            no mass to normalise by.
    """
    response = np.asarray(response_map, dtype=np.float64)
    if response.size != grid_shape[0] * grid_shape[1]:
        raise ValueError("the response does not fill the grid shape it was given")
    kernel = np.clip(response.reshape(grid_shape), 0.0, None)
    total = float(np.sum(kernel))
    if total <= DENSITY_FLOOR:
        raise ValueError("the response carries no mass, so it cannot be normalised")
    centred = np.fft.ifftshift(kernel / total)
    return PointSpreadOperator(
        kernel_spectrum=np.fft.rfft2(centred),
        grid_shape=grid_shape,
        grid_spacing_m=grid_spacing_m,
        normalisation=total,
    )


def apply_operator(
    operator: PointSpreadOperator, density: Float64Array
) -> Float64Array:
    """Blurs a source density into the map it would produce.

    Args:
        operator: The response operator.
        density: Source density, shape `(n_y, n_x)`.

    Returns:
        The blurred map, same shape.
    """
    spectrum = np.fft.rfft2(np.asarray(density, dtype=np.float64))
    return np.asarray(
        np.fft.irfft2(spectrum * operator.kernel_spectrum, operator.grid_shape),
        dtype=np.float64,
    )


def apply_operator_transpose(
    operator: PointSpreadOperator, residual: Float64Array
) -> Float64Array:
    """Applies the operator's adjoint, which for a real response is its mirror.

    Args:
        operator: The response operator.
        residual: A map-shaped array, shape `(n_y, n_x)`.

    Returns:
        The correlated array, same shape.
    """
    spectrum = np.fft.rfft2(np.asarray(residual, dtype=np.float64))
    return np.asarray(
        np.fft.irfft2(
            spectrum * np.conj(operator.kernel_spectrum), operator.grid_shape
        ),
        dtype=np.float64,
    )


def deconvolve_richardson_lucy(
    map_values: Float64Array,
    operator: PointSpreadOperator,
    iteration_count: int,
) -> Float64Array:
    """Recovers a non-negative source density by Richardson-Lucy.

    The update is multiplicative, so a density that starts non-negative stays
    non-negative without any projection — which is what makes it the right
    starting point for a quantity that cannot be negative.

    There is no natural stopping point: the iteration sharpens the estimate and
    amplifies noise monotonically, so the count is a parameter and belongs in
    configuration rather than being tuned by eye.

    Args:
        map_values: The prepared map, shape `(n_cells,)` or `(n_y, n_x)`.
        operator: The response operator.
        iteration_count: How many updates to apply.

    Returns:
        The source density, shape `(n_y, n_x)`, non-negative.

    Raises:
        ValueError: If fewer than one iteration is requested.
    """
    if iteration_count < 1:
        raise ValueError("Richardson-Lucy needs at least one iteration")
    observed = np.clip(
        np.asarray(map_values, dtype=np.float64).reshape(operator.grid_shape), 0.0, None
    )
    density = np.full(operator.grid_shape, float(np.mean(observed)) + DENSITY_FLOOR)
    for _ in range(iteration_count):
        blurred = np.clip(apply_operator(operator, density), DENSITY_FLOOR, None)
        density = density * apply_operator_transpose(operator, observed / blurred)
        density = np.clip(density, 0.0, None)
    return np.asarray(density, dtype=np.float64)


def compute_total_variation_drag(density: Float64Array) -> Float64Array:
    """Divergence of the density's normalised gradient.

    This is the term that pulls a total-variation-regularised update towards
    piecewise-constant regions with sharp edges between them, which is the
    shape a radiating perimeter around a silent interior actually has.

    Args:
        density: Source density, shape `(n_y, n_x)`.

    Returns:
        The divergence, same shape.
    """
    values = np.asarray(density, dtype=np.float64)
    gradient_y, gradient_x = np.gradient(values)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2) + GRADIENT_FLOOR
    divergence_x = np.gradient(gradient_x / magnitude, axis=1)
    divergence_y = np.gradient(gradient_y / magnitude, axis=0)
    return np.asarray(divergence_x + divergence_y, dtype=np.float64)


def deconvolve_with_total_variation(
    map_values: Float64Array,
    operator: PointSpreadOperator,
    regularization_weight: float,
    iteration_count: int,
) -> Float64Array:
    """Richardson-Lucy with a total-variation prior on the density.

    Args:
        map_values: The prepared map, shape `(n_cells,)` or `(n_y, n_x)`.
        operator: The response operator.
        regularization_weight: Strength of the prior. Zero recovers plain
            Richardson-Lucy exactly.
        iteration_count: How many updates to apply.

    Returns:
        The source density, shape `(n_y, n_x)`, non-negative.

    Raises:
        ValueError: If fewer than one iteration is requested, or the weight is
            negative.
    """
    if iteration_count < 1:
        raise ValueError("Richardson-Lucy needs at least one iteration")
    if regularization_weight < 0.0:
        raise ValueError("a regularisation weight cannot be negative")
    observed = np.clip(
        np.asarray(map_values, dtype=np.float64).reshape(operator.grid_shape), 0.0, None
    )
    density = np.full(operator.grid_shape, float(np.mean(observed)) + DENSITY_FLOOR)
    for _ in range(iteration_count):
        blurred = np.clip(apply_operator(operator, density), DENSITY_FLOOR, None)
        update = apply_operator_transpose(operator, observed / blurred)
        drag = 1.0 - regularization_weight * compute_total_variation_drag(density)
        density = density * update / np.clip(drag, 0.25, 4.0)
        density = np.clip(density, 0.0, None)
    return np.asarray(density, dtype=np.float64)


def compute_l_curve(
    map_values: Float64Array,
    operator: PointSpreadOperator,
    regularization_weights: Float64Array,
    iteration_count: int,
) -> tuple[Float64Array, Float64Array]:
    """Data misfit against solution roughness, over a range of weights.

    The corner of the resulting curve is where tightening the prior stops
    buying smoothness and starts costing fit. Recording the curve is what makes
    the chosen weight a reading rather than a preference.

    Args:
        map_values: The prepared map.
        operator: The response operator.
        regularization_weights: Weights to evaluate, shape `(n_weights,)`.
        iteration_count: Iterations per weight.

    Returns:
        `(data_misfits, roughnesses)`, both shape `(n_weights,)`.
    """
    observed = np.clip(
        np.asarray(map_values, dtype=np.float64).reshape(operator.grid_shape), 0.0, None
    )
    misfits = np.empty(len(regularization_weights), dtype=np.float64)
    roughnesses = np.empty(len(regularization_weights), dtype=np.float64)
    for weight_index, weight in enumerate(regularization_weights):
        density = deconvolve_with_total_variation(
            observed, operator, float(weight), iteration_count
        )
        misfits[weight_index] = float(
            np.linalg.norm(apply_operator(operator, density) - observed)
        )
        gradient_y, gradient_x = np.gradient(density)
        roughnesses[weight_index] = float(
            np.sum(np.sqrt(gradient_x**2 + gradient_y**2))
        )
    return misfits, roughnesses
