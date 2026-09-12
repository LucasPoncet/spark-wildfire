"""Recovering a source density from the map it produced.

The annulus case is the plan's own gate criterion and the one that matters:
a ring convolved with a known response, put back through the deconvolution,
has to come out at the right radius. It is the whole premise of reading a front
off a map that cannot resolve it.

The operator tests exist because a transform-based convolution has two easy
ways to be silently wrong — an unshifted kernel, which displaces everything by
half the grid, and a transpose that is not the adjoint, which makes
Richardson-Lucy converge to the wrong thing rather than fail.
"""

import numpy as np
import pytest

from src.spark.inverse.map_deconvolution import (
    PointSpreadOperator,
    apply_operator,
    apply_operator_transpose,
    build_point_spread_operator,
    compute_l_curve,
    compute_total_variation_drag,
    deconvolve_richardson_lucy,
    deconvolve_with_total_variation,
)

GRID_SIZE: int = 96
GRID_SPACING_M: float = 0.5
GRID_SHAPE: tuple[int, int] = (GRID_SIZE, GRID_SIZE)
CENTRE: float = (GRID_SIZE - 1) / 2.0
RESPONSE_WIDTH_CELLS: float = 3.0


def build_axes() -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(GRID_SIZE, dtype=np.float64)
    return np.meshgrid(index - CENTRE, index - CENTRE)


def build_response() -> np.ndarray:
    offset_x, offset_y = build_axes()
    return np.asarray(
        np.exp(-0.5 * (offset_x**2 + offset_y**2) / RESPONSE_WIDTH_CELLS**2).ravel(),
        dtype=np.float64,
    )


def build_annulus(radius_cells: float, width_cells: float) -> np.ndarray:
    offset_x, offset_y = build_axes()
    radii = np.sqrt(offset_x**2 + offset_y**2)
    return np.asarray(
        np.exp(-0.5 * ((radii - radius_cells) / width_cells) ** 2), dtype=np.float64
    )


def measure_ring_radius_cells(density: np.ndarray) -> float:
    offset_x, offset_y = build_axes()
    radii = np.sqrt(offset_x**2 + offset_y**2)
    weights = np.clip(density, 0.0, None)
    total = float(np.sum(weights))
    return float(np.sum(weights * radii) / total) if total > 0.0 else 0.0


def build_operator() -> PointSpreadOperator:
    return build_point_spread_operator(build_response(), GRID_SHAPE, GRID_SPACING_M)


def test_a_point_source_is_blurred_into_the_response_itself() -> None:
    operator = build_operator()
    point = np.zeros(GRID_SHAPE)
    point[GRID_SIZE // 2, GRID_SIZE // 2] = 1.0
    blurred = apply_operator(operator, point)
    peak_row, peak_column = np.unravel_index(int(np.argmax(blurred)), GRID_SHAPE)
    assert abs(int(peak_row) - GRID_SIZE // 2) <= 1
    assert abs(int(peak_column) - GRID_SIZE // 2) <= 1


def test_the_operator_conserves_mass() -> None:
    operator = build_operator()
    density = build_annulus(12.0, 2.0)
    assert float(np.sum(apply_operator(operator, density))) == pytest.approx(
        float(np.sum(density)), rel=1e-9
    )


def test_the_transpose_is_the_adjoint() -> None:
    """`<A x, y> == <x, A* y>`, which Richardson-Lucy relies on to converge."""
    operator = build_operator()
    generator = np.random.default_rng(0)
    first = generator.random(GRID_SHAPE)
    second = generator.random(GRID_SHAPE)
    assert float(np.sum(apply_operator(operator, first) * second)) == pytest.approx(
        float(np.sum(first * apply_operator_transpose(operator, second))), rel=1e-9
    )


def test_a_response_that_does_not_fill_its_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not fill the grid shape"):
        build_point_spread_operator(np.ones(10), GRID_SHAPE, GRID_SPACING_M)


def test_an_empty_response_is_rejected() -> None:
    with pytest.raises(ValueError, match="carries no mass"):
        build_point_spread_operator(
            np.zeros(GRID_SIZE * GRID_SIZE), GRID_SHAPE, GRID_SPACING_M
        )


@pytest.mark.parametrize("radius_cells", [10.0, 16.0])
def test_a_blurred_annulus_is_recovered_at_its_own_radius(radius_cells: float) -> None:
    operator = build_operator()
    truth = build_annulus(radius_cells, 1.5)
    observed = apply_operator(operator, truth)
    density = deconvolve_richardson_lucy(observed, operator, 60)
    recovered_m = measure_ring_radius_cells(density) * GRID_SPACING_M
    assert recovered_m == pytest.approx(
        measure_ring_radius_cells(truth) * GRID_SPACING_M, abs=0.5
    )


def test_deconvolution_sharpens_what_the_response_blurred() -> None:
    operator = build_operator()
    truth = build_annulus(14.0, 1.5)
    observed = apply_operator(operator, truth)
    density = deconvolve_richardson_lucy(observed, operator, 60)
    assert float(np.max(density)) > float(np.max(observed))


def test_the_recovered_density_is_never_negative() -> None:
    operator = build_operator()
    observed = apply_operator(operator, build_annulus(12.0, 2.0))
    assert np.all(deconvolve_richardson_lucy(observed, operator, 40) >= 0.0)


def test_no_iterations_is_rejected() -> None:
    operator = build_operator()
    with pytest.raises(ValueError, match="at least one iteration"):
        deconvolve_richardson_lucy(np.ones(GRID_SHAPE), operator, 0)


def test_a_zero_weight_recovers_plain_richardson_lucy() -> None:
    operator = build_operator()
    observed = apply_operator(operator, build_annulus(12.0, 2.0))
    assert deconvolve_with_total_variation(
        observed, operator, 0.0, 30
    ) == pytest.approx(deconvolve_richardson_lucy(observed, operator, 30))


def test_a_negative_regularisation_weight_is_rejected() -> None:
    operator = build_operator()
    with pytest.raises(ValueError, match="cannot be negative"):
        deconvolve_with_total_variation(np.ones(GRID_SHAPE), operator, -1.0, 10)


def test_total_variation_smooths_a_noisy_recovery() -> None:
    operator = build_operator()
    generator = np.random.default_rng(1)
    observed = apply_operator(operator, build_annulus(12.0, 2.0))
    observed = np.clip(observed + 0.002 * generator.random(GRID_SHAPE), 0.0, None)
    plain = deconvolve_richardson_lucy(observed, operator, 60)
    regularised = deconvolve_with_total_variation(observed, operator, 0.05, 60)

    def roughness(density: np.ndarray) -> float:
        gradient_y, gradient_x = np.gradient(density / np.max(density))
        return float(np.sum(np.sqrt(gradient_x**2 + gradient_y**2)))

    assert roughness(regularised) < roughness(plain)


def test_the_drag_term_is_zero_on_a_flat_density() -> None:
    assert compute_total_variation_drag(np.ones(GRID_SHAPE)) == pytest.approx(
        np.zeros(GRID_SHAPE), abs=1e-9
    )


def test_the_l_curve_trades_misfit_against_roughness() -> None:
    operator = build_operator()
    observed = apply_operator(operator, build_annulus(12.0, 2.0))
    misfits, roughnesses = compute_l_curve(
        observed, operator, np.array([0.0, 0.05, 0.2]), 25
    )
    assert misfits.size == 3
    assert roughnesses[0] >= roughnesses[-1]


ODD_SIZE: int = 97
ODD_SHAPE: tuple[int, int] = (ODD_SIZE, ODD_SIZE)
ODD_CENTRE: int = ODD_SIZE // 2


def build_odd_axes() -> tuple[np.ndarray, np.ndarray]:
    """An odd grid, so one cell sits exactly on the centre and none tie."""
    index = np.arange(ODD_SIZE, dtype=np.float64) - ODD_CENTRE
    return np.meshgrid(index, index)


def build_odd_response(shift_cells: float) -> np.ndarray:
    """The response, evaluated `shift_cells` from the grid centre.

    Which is what a caller imaging a fire actually has: the response is
    computed at the source being imaged, not at the middle of the domain.
    """
    offset_x, offset_y = build_odd_axes()
    return np.asarray(
        np.exp(
            -0.5
            * ((offset_x - shift_cells) ** 2 + offset_y**2)
            / RESPONSE_WIDTH_CELLS**2
        ).ravel(),
        dtype=np.float64,
    )


def build_odd_annulus(radius_cells: float, width_cells: float) -> np.ndarray:
    offset_x, offset_y = build_odd_axes()
    radii = np.sqrt(offset_x**2 + offset_y**2)
    return np.asarray(
        np.exp(-0.5 * ((radii - radius_cells) / width_cells) ** 2), dtype=np.float64
    )


def measure_odd_centroid_cells(density: np.ndarray) -> tuple[float, float]:
    offset_x, offset_y = build_odd_axes()
    weights = np.clip(density, 0.0, None)
    total = float(np.sum(weights))
    return (
        float(np.sum(weights * offset_x) / total),
        float(np.sum(weights * offset_y) / total),
    )


def measure_odd_ring_radius_cells(density: np.ndarray) -> float:
    offset_x, offset_y = build_odd_axes()
    radii = np.sqrt(offset_x**2 + offset_y**2)
    weights = np.clip(density, 0.0, None)
    return float(np.sum(weights * radii) / np.sum(weights))


@pytest.mark.parametrize("shift_cells", [0, 4, 12, 20])
def test_a_response_blurs_in_place_wherever_it_was_evaluated(shift_cells: int) -> None:
    """A kernel carries its own offset, and that offset is not the density's."""
    operator = build_point_spread_operator(
        build_odd_response(shift_cells), ODD_SHAPE, GRID_SPACING_M
    )
    source = np.zeros(ODD_SHAPE)
    source[ODD_CENTRE, ODD_CENTRE] = 1.0
    centre_x_cells, centre_y_cells = measure_odd_centroid_cells(
        apply_operator(operator, source)
    )
    assert centre_x_cells == pytest.approx(0.0, abs=1e-6)
    assert centre_y_cells == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("shift_cells", [4, 12, 20])
def test_an_off_centre_response_does_not_displace_what_it_recovers(
    shift_cells: int,
) -> None:
    """The defect this centring exists to stop.

    The map is blurred by the response as it physically acts, in place. The
    operator is built from a response evaluated somewhere else, as every caller
    imaging a moving fire must. Without re-centring the recovered density comes
    back displaced by that separation, and a front read off it is wrong by the
    distance the fire has travelled.
    """
    observed = apply_operator(
        build_point_spread_operator(build_odd_response(0), ODD_SHAPE, GRID_SPACING_M),
        build_odd_annulus(12.0, 1.5),
    )
    recovered = deconvolve_with_total_variation(
        observed,
        build_point_spread_operator(
            build_odd_response(shift_cells), ODD_SHAPE, GRID_SPACING_M
        ),
        0.01,
        40,
    )
    centre_x_cells, centre_y_cells = measure_odd_centroid_cells(recovered)
    assert centre_x_cells == pytest.approx(0.0, abs=0.5)
    assert centre_y_cells == pytest.approx(0.0, abs=0.5)
    assert measure_odd_ring_radius_cells(recovered) == pytest.approx(12.0, abs=1.0)


def test_the_operator_does_not_depend_on_where_the_response_was_evaluated() -> None:
    density = build_odd_annulus(10.0, 2.0)
    assert apply_operator(
        build_point_spread_operator(build_odd_response(15), ODD_SHAPE, GRID_SPACING_M),
        density,
    ) == pytest.approx(
        apply_operator(
            build_point_spread_operator(
                build_odd_response(0), ODD_SHAPE, GRID_SPACING_M
            ),
            density,
        ),
        abs=1e-9,
    )


@pytest.mark.parametrize("shift_cells", [0.25, 0.37, 4.37, 20.8])
def test_a_sub_cell_response_offset_is_taken_out_by_the_phase_ramp(
    shift_cells: float,
) -> None:
    """What the roll alone cannot do, since it moves whole cells only.

    A roll leaves up to half a cell — 0.25 m at the imaging spacing, which is
    the same order as the final-frame radial error it would then sit inside.
    The phase ramp takes the residue to under a hundredth of a cell.
    """
    operator = build_point_spread_operator(
        build_odd_response(shift_cells), ODD_SHAPE, GRID_SPACING_M
    )
    source = np.zeros(ODD_SHAPE)
    source[ODD_CENTRE, ODD_CENTRE] = 1.0
    centre_x_cells, centre_y_cells = measure_odd_centroid_cells(
        apply_operator(operator, source)
    )
    assert centre_x_cells == pytest.approx(0.0, abs=0.02)
    assert centre_y_cells == pytest.approx(0.0, abs=0.02)


def test_the_sub_cell_correction_leaves_the_mass_alone() -> None:
    """The ramp touches every frequency but the zero one, so mass is untouched."""
    density = build_odd_annulus(10.0, 2.0)
    blurred = apply_operator(
        build_point_spread_operator(
            build_odd_response(0.37), ODD_SHAPE, GRID_SPACING_M
        ),
        density,
    )
    assert float(np.sum(blurred)) == pytest.approx(float(np.sum(density)))


def test_the_transpose_is_still_the_adjoint_under_a_sub_cell_shift() -> None:
    """A ramp that broke the adjoint would make Richardson-Lucy converge wrong."""
    operator = build_point_spread_operator(
        build_odd_response(0.37), ODD_SHAPE, GRID_SPACING_M
    )
    generator = np.random.default_rng(11)
    left = generator.normal(size=ODD_SHAPE)
    right = generator.normal(size=ODD_SHAPE)
    assert float(np.sum(right * apply_operator(operator, left))) == pytest.approx(
        float(np.sum(left * apply_operator_transpose(operator, right))), rel=1e-9
    )


def test_a_sub_cell_offset_response_recovers_a_ring_in_place() -> None:
    observed = apply_operator(
        build_point_spread_operator(build_odd_response(0), ODD_SHAPE, GRID_SPACING_M),
        build_odd_annulus(12.0, 1.5),
    )
    recovered = deconvolve_with_total_variation(
        observed,
        build_point_spread_operator(build_odd_response(8.4), ODD_SHAPE, GRID_SPACING_M),
        0.01,
        40,
    )
    centre_x_cells, centre_y_cells = measure_odd_centroid_cells(recovered)
    assert centre_x_cells == pytest.approx(0.0, abs=0.2)
    assert centre_y_cells == pytest.approx(0.0, abs=0.2)
    assert measure_odd_ring_radius_cells(recovered) == pytest.approx(12.0, abs=1.0)
