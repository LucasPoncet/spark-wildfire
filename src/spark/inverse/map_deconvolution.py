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

**The re-centring in that measurement is also required of the operator
itself**, and leaving it out was the largest error in the front position tier.
A convolution kernel carries its own offset. Callers evaluate the response at
the source they are imaging rather than at the middle of the domain, so a
kernel built straight from it translates by the separation between the two,
and the density it recovers comes back displaced by the same amount in the
opposite direction. On `configs/f1` that separation is the distance the fire
has travelled: 0.71 m at five seconds, 8.02 m at fifty. The free-form contour's
mean radial error tracked it — 0.35 m to 6.66 m — and so did the parametric
perimeter's, 0.48 m to 7.01 m, because both routes share this operator. Both
looked like a model that degrades as a front outgrows the array's resolution.
Neither was. With the kernel centred the final-frame radial error is 0.24 m.

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

    The response is moved onto the grid centre before its transform is taken,
    and that step is not cosmetic. A convolution kernel carries its own offset:
    one whose peak sits `d` from the grid centre convolves *and* translates by
    `d`, so a deconvolution built from it hands back a density displaced by
    `-d`. Callers evaluate the response at the source they are imaging rather
    than at the middle of the domain — which is the accurate thing to do for
    its shape — so the offset is whatever the fire has drifted, and it grows
    with the fire. Re-centring here keeps the shape and drops the translation.

    Centring is done in two steps because a roll can only move whole cells. The
    roll puts the response's brightest cell on the grid centre, and a linear
    phase ramp on the spectrum takes out the fraction of a cell left over — a
    shift in the transform domain is a multiplication, so the sub-cell part
    costs one complex array and no resampling. The remaining offset is then set
    by how well the peak's own curvature locates it rather than by the grid,
    which at the imaging spacing takes 0.25 m down to under a centimetre.

    Args:
        response_map: The response to a point source, shape `(n_cells,)`. It
            may have been evaluated anywhere on the grid; its peak is taken to
            mark where that source was.
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
    rolled = _move_peak_to_grid_centre(kernel / total)
    spectrum = np.fft.rfft2(np.fft.ifftshift(rolled))
    return PointSpreadOperator(
        kernel_spectrum=spectrum
        * _build_phase_ramp(grid_shape, _estimate_sub_cell_offset_cells(rolled)),
        grid_shape=grid_shape,
        grid_spacing_m=grid_spacing_m,
        normalisation=total,
    )


def _move_peak_to_grid_centre(kernel: Float64Array) -> Float64Array:
    """Rolls a response so its brightest cell sits at the grid centre.

    Args:
        kernel: The response, shape `(n_y, n_x)`.

    Returns:
        The same response, rolled. Unchanged when it is already centred.
    """
    peak_row, peak_column = np.unravel_index(int(np.argmax(kernel)), kernel.shape)
    centre_row = kernel.shape[0] // 2
    centre_column = kernel.shape[1] // 2
    return np.asarray(
        np.roll(
            kernel,
            (centre_row - int(peak_row), centre_column - int(peak_column)),
            axis=(0, 1),
        ),
        dtype=np.float64,
    )


def _estimate_sub_cell_offset_cells(kernel: Float64Array) -> Float64Array:
    """Where the response's true centre sits inside its brightest cell.

    Read off the curvature through the peak, one axis at a time: three samples
    fix a parabola and its vertex is the estimate. A peak on a flat or rising
    run has no vertex to find, and that axis reports zero rather than a
    division by a vanishing second difference.

    Args:
        kernel: The rolled response, shape `(n_y, n_x)`, peak at the centre.

    Returns:
        `(row_offset, column_offset)` in cells, each within half a cell.
    """
    peak_row = kernel.shape[0] // 2
    peak_column = kernel.shape[1] // 2
    return np.array(
        [
            _locate_parabola_vertex(kernel[peak_row - 1 : peak_row + 2, peak_column]),
            _locate_parabola_vertex(
                kernel[peak_row, peak_column - 1 : peak_column + 2]
            ),
        ],
        dtype=np.float64,
    )


def _locate_parabola_vertex(samples: Float64Array) -> float:
    """Vertex of the parabola through three samples, relative to the middle.

    Args:
        samples: Three consecutive samples straddling a maximum.

    Returns:
        The vertex offset in samples, clipped to half a sample either way.
    """
    if samples.size != 3:
        return 0.0
    before, middle, after = (float(value) for value in samples)
    curvature = before - 2.0 * middle + after
    if abs(curvature) <= GRADIENT_FLOOR:
        return 0.0
    return float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))


def _build_phase_ramp(
    grid_shape: tuple[int, int], offset_cells: Float64Array
) -> ComplexArray:
    """The spectrum of a shift, for undoing a sub-cell offset.

    Translating by `s` samples multiplies a spectrum by `exp(-2i*pi*f*s)`, so
    undoing an offset of `d` means multiplying by `exp(+2i*pi*f*d)`. The zero
    frequency is untouched, which is why this leaves the operator's mass and
    its adjoint alone.

    Args:
        grid_shape: `(n_y, n_x)` of the grid.
        offset_cells: `(row_offset, column_offset)` to undo, in cells.

    Returns:
        A half-spectrum-shaped array of unit-modulus factors.
    """
    row_frequencies = np.fft.fftfreq(grid_shape[0])
    column_frequencies = np.fft.rfftfreq(grid_shape[1])
    phase = (
        2.0
        * np.pi
        * (
            row_frequencies[:, None] * float(offset_cells[0])
            + column_frequencies[None, :] * float(offset_cells[1])
        )
    )
    return np.asarray(np.exp(1j * phase), dtype=np.complex128)


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
