"""Reading a front distance off one line through the map.

The one-sided test is the point of the module. The plan's model assumes a front
that radiates both ahead of and behind the ignition point; a fire that burns
out behind itself does not, and forcing a trailing lobe would make the head
distance absorb the error. What is asserted is that the fitted ratio reports
which case it is in.
"""

import numpy as np
import pytest

from src.spark.inverse.front_radial_profile import (
    FrontBandEstimate,
    FrontDistanceEstimate,
    build_band_model,
    build_grid_axes,
    build_two_lobe_model,
    estimate_front_band_by_matched_filter,
    estimate_front_distance_by_matched_filter,
    extract_radial_profile,
    sample_map_bilinear,
)

GRID_SPACING_M: float = 0.25
DOMAIN_EXTENT_M: float = 40.0
CENTRE_XY_M = np.array([20.0, 20.0])
RESPONSE_WIDTH_M: float = 1.2
MAXIMUM_RADIUS_M: float = 16.0
RADIUS_STEP_M: float = 0.1


def build_grid() -> np.ndarray:
    axis_m = np.arange(0.5 * GRID_SPACING_M, DOMAIN_EXTENT_M, GRID_SPACING_M)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    return np.stack(
        (grid_x_m.ravel(), grid_y_m.ravel(), np.full(grid_x_m.size, 0.5)), axis=1
    )


def build_lobes(
    positions_xyz_m: np.ndarray,
    head_distance_m: float,
    back_distance_m: float,
    back_to_head_ratio: float,
) -> np.ndarray:
    offsets_m = positions_xyz_m[:, :2] - CENTRE_XY_M
    head = np.exp(
        -0.5
        * np.sum((offsets_m - np.array([head_distance_m, 0.0])) ** 2, axis=1)
        / RESPONSE_WIDTH_M**2
    )
    back = np.exp(
        -0.5
        * np.sum((offsets_m + np.array([back_distance_m, 0.0])) ** 2, axis=1)
        / RESPONSE_WIDTH_M**2
    )
    return np.asarray(head + back_to_head_ratio * back, dtype=np.float64)


def build_response_profile() -> tuple[np.ndarray, np.ndarray]:
    radii_m = np.arange(
        -MAXIMUM_RADIUS_M, MAXIMUM_RADIUS_M + RADIUS_STEP_M, RADIUS_STEP_M
    )
    return radii_m, np.exp(-0.5 * (radii_m / RESPONSE_WIDTH_M) ** 2)


def fit(map_values: np.ndarray) -> FrontDistanceEstimate:
    radii_m, profile = extract_radial_profile(
        map_values,
        build_grid(),
        CENTRE_XY_M,
        np.array([1.0, 0.0]),
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    response_radii_m, response_profile = build_response_profile()
    return estimate_front_distance_by_matched_filter(
        radii_m, profile, response_radii_m, response_profile, MAXIMUM_RADIUS_M
    )


def test_the_grid_axes_come_back_ascending_and_unique() -> None:
    x_m, y_m = build_grid_axes(build_grid())
    assert np.all(np.diff(x_m) > 0.0)
    assert x_m.size * y_m.size == build_grid().shape[0]


def test_sampling_at_a_cell_centre_returns_that_cell() -> None:
    positions_xyz_m = build_grid()
    values = build_lobes(positions_xyz_m, 5.0, 3.0, 0.5)
    index = 5000
    sampled = sample_map_bilinear(
        values, positions_xyz_m, positions_xyz_m[index : index + 1, :2]
    )
    assert float(sampled[0]) == pytest.approx(float(values[index]), rel=1e-6)


def test_sampling_outside_the_grid_is_clamped_not_extrapolated() -> None:
    positions_xyz_m = build_grid()
    values = build_lobes(positions_xyz_m, 5.0, 3.0, 0.5)
    far = sample_map_bilinear(values, positions_xyz_m, np.array([[500.0, 500.0]]))
    assert np.isfinite(far[0])


def test_a_profile_runs_both_ways_from_its_origin() -> None:
    radii_m, profile = extract_radial_profile(
        build_lobes(build_grid(), 5.0, 3.0, 0.5),
        build_grid(),
        CENTRE_XY_M,
        np.array([1.0, 0.0]),
        MAXIMUM_RADIUS_M,
        RADIUS_STEP_M,
    )
    assert float(radii_m[0]) == pytest.approx(-MAXIMUM_RADIUS_M)
    assert float(radii_m[-1]) == pytest.approx(MAXIMUM_RADIUS_M, abs=RADIUS_STEP_M)
    assert profile.size == radii_m.size


def test_a_zero_length_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero length"):
        extract_radial_profile(
            build_lobes(build_grid(), 5.0, 3.0, 0.5),
            build_grid(),
            CENTRE_XY_M,
            np.zeros(2),
            MAXIMUM_RADIUS_M,
            RADIUS_STEP_M,
        )


def test_a_non_positive_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="must both be positive"):
        extract_radial_profile(
            build_lobes(build_grid(), 5.0, 3.0, 0.5),
            build_grid(),
            CENTRE_XY_M,
            np.array([1.0, 0.0]),
            MAXIMUM_RADIUS_M,
            0.0,
        )


def test_the_model_peaks_where_its_lobes_are_placed() -> None:
    radii_m, response_profile = build_response_profile()
    model = build_two_lobe_model(radii_m, radii_m, response_profile, 6.0, 3.0, 0.0)
    assert float(radii_m[int(np.argmax(model))]) == pytest.approx(6.0, abs=0.2)


@pytest.mark.parametrize("head_distance_m", [4.0, 7.0, 10.0])
def test_a_two_lobed_map_gives_back_its_head_distance(head_distance_m: float) -> None:
    estimate = fit(build_lobes(build_grid(), head_distance_m, 3.0, 0.6))
    assert estimate.head_distance_m == pytest.approx(head_distance_m, abs=0.3)


def test_a_two_lobed_map_gives_back_its_back_distance() -> None:
    estimate = fit(build_lobes(build_grid(), 7.0, 3.5, 0.6))
    assert estimate.back_distance_m == pytest.approx(3.5, abs=0.4)
    assert not estimate.is_one_sided


def test_a_map_with_no_trailing_lobe_is_reported_as_one_sided() -> None:
    """The condition the plan's two-lobe model does not anticipate."""
    estimate = fit(build_lobes(build_grid(), 8.0, 3.0, 0.0))
    assert estimate.is_one_sided
    assert estimate.back_to_head_ratio < 0.15


def test_a_one_sided_map_still_gives_the_right_head_distance() -> None:
    estimate = fit(build_lobes(build_grid(), 8.0, 3.0, 0.0))
    assert estimate.head_distance_m == pytest.approx(8.0, abs=0.3)


def test_the_fit_reports_a_small_residual_on_a_map_it_explains() -> None:
    assert fit(build_lobes(build_grid(), 7.0, 3.0, 0.5)).residual < 0.05


def test_an_empty_profile_is_rejected() -> None:
    positions_xyz_m = build_grid()
    with pytest.raises(ValueError, match="carries no signal"):
        fit(np.zeros(positions_xyz_m.shape[0]))


def build_band_profile(
    inner_edge_m: float, outer_edge_m: float, response_width_m: float
) -> tuple[np.ndarray, np.ndarray]:
    radii_m = np.arange(
        -MAXIMUM_RADIUS_M, MAXIMUM_RADIUS_M + RADIUS_STEP_M, RADIUS_STEP_M
    )
    band = ((radii_m >= inner_edge_m) & (radii_m <= outer_edge_m)).astype(np.float64)
    kernel = np.exp(-0.5 * (radii_m / response_width_m) ** 2)
    return radii_m, np.convolve(band, kernel / kernel.sum(), mode="same")


def fit_band(
    inner_edge_m: float, outer_edge_m: float, response_width_m: float
) -> FrontBandEstimate:
    radii_m, profile = build_band_profile(inner_edge_m, outer_edge_m, response_width_m)
    response_profile = np.exp(-0.5 * (radii_m / response_width_m) ** 2)
    return estimate_front_band_by_matched_filter(
        radii_m, profile, radii_m, response_profile, MAXIMUM_RADIUS_M
    )


@pytest.mark.parametrize("inner_edge_m", [2.0, 5.0, 8.0, 11.0])
def test_the_band_model_recovers_the_leading_edge(inner_edge_m: float) -> None:
    """What a head distance actually means, and what the lobe fit does not give."""
    estimate = fit_band(inner_edge_m, 12.0, 1.2)
    assert estimate.outer_edge_m == pytest.approx(12.0, abs=0.2)


@pytest.mark.parametrize("response_width_m", [0.8, 1.5, 2.5])
def test_the_leading_edge_survives_any_response_width(response_width_m: float) -> None:
    estimate = fit_band(5.0, 12.0, response_width_m)
    assert estimate.outer_edge_m == pytest.approx(12.0, abs=0.2)


def test_the_band_model_recovers_both_edges() -> None:
    estimate = fit_band(5.0, 12.0, 1.2)
    assert estimate.inner_edge_m == pytest.approx(5.0, abs=0.2)
    assert estimate.half_width_m == pytest.approx(3.5, abs=0.2)


def test_a_single_lobe_fit_falls_short_by_the_band_half_width() -> None:
    """The measurement that sent the head distance to a band model.

    A one-lobe fit lands on the band's centre, so its shortfall to the leading
    edge is the band's own half-width — and that is true whatever the response
    is, which is what rules out correcting it with a response width.
    """
    radii_m, profile = build_band_profile(5.0, 12.0, 1.2)
    response_profile = np.exp(-0.5 * (radii_m / 1.2) ** 2)
    lobe = estimate_front_distance_by_matched_filter(
        radii_m, profile, radii_m, response_profile, MAXIMUM_RADIUS_M
    )
    assert 12.0 - lobe.head_distance_m == pytest.approx(3.5, abs=0.3)


def test_the_band_model_is_normalised_to_its_own_peak() -> None:
    radii_m = np.arange(
        -MAXIMUM_RADIUS_M, MAXIMUM_RADIUS_M + RADIUS_STEP_M, RADIUS_STEP_M
    )
    response_profile = np.exp(-0.5 * (radii_m / 1.2) ** 2)
    model = build_band_model(radii_m, radii_m, response_profile, 5.0, 12.0)
    assert float(np.max(model)) == pytest.approx(1.0)


def test_an_empty_profile_is_rejected_by_the_band_fit() -> None:
    radii_m = np.arange(
        -MAXIMUM_RADIUS_M, MAXIMUM_RADIUS_M + RADIUS_STEP_M, RADIUS_STEP_M
    )
    with pytest.raises(ValueError, match="carries no signal"):
        estimate_front_band_by_matched_filter(
            radii_m, np.zeros(radii_m.size), radii_m, np.ones(radii_m.size), 10.0
        )
