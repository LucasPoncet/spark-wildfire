"""Fitting head, back, centroid and expansion rates against time.

The closure identity is pinned here for what it is rather than what the plan
claims of it. Head and back are both built from one centroid and one semi-axis,
so their difference is twice that semi-axis no matter what the bearing is or
how the shape factor scales it — both substitutions leave the identity exactly
satisfied, and the tests below demonstrate each. The residual measures
departure from constant rates and nothing else.
"""

import numpy as np
import pytest

from src.spark.inverse.fire_extent import ExtentEstimate
from src.spark.inverse.fire_rate_of_spread import (
    estimate_rate_of_spread,
    project_head_and_back_m,
)

BOOTSTRAP_COUNT: int = 60
CENTRE_XY_M = np.array([30.0, 30.0])


def build_extent(
    time_s: float,
    centroid_xy_m: np.ndarray,
    semi_axis_major_m: float,
    is_resolved: bool = True,
) -> ExtentEstimate:
    return ExtentEstimate(
        time_s=time_s,
        centroid_xy_m=centroid_xy_m,
        semi_axis_major_m=semi_axis_major_m,
        semi_axis_minor_m=0.5 * semi_axis_major_m,
        orientation_rad=0.0,
        is_resolved=is_resolved,
        upper_bound_only=not is_resolved,
        observed_semi_axis_major_m=semi_axis_major_m + 1.0,
    )


def build_series(
    bearing_rad: float,
    centroid_speed_m_per_s: float,
    expansion_rate_m_per_s: float,
    frame_count: int = 10,
) -> list[ExtentEstimate]:
    direction = np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
    times_s = np.arange(1, frame_count + 1, dtype=np.float64) * 5.0
    return [
        build_extent(
            float(time_s),
            CENTRE_XY_M + centroid_speed_m_per_s * time_s * direction,
            1.0 + expansion_rate_m_per_s * float(time_s),
        )
        for time_s in times_s
    ]


@pytest.mark.parametrize("bearing_deg", [0.0, 37.0, 145.0, -80.0])
def test_a_linear_series_gives_back_the_rates_it_was_built_from(
    bearing_deg: float,
) -> None:
    bearing_rad = np.deg2rad(bearing_deg)
    estimate = estimate_rate_of_spread(
        build_series(bearing_rad, 0.19, 0.10), bearing_rad, 5, BOOTSTRAP_COUNT, 0
    )
    assert estimate.centroid_speed_m_per_s == pytest.approx(0.19, rel=1e-6)
    assert estimate.expansion_rate_m_per_s == pytest.approx(0.10, rel=1e-6)
    assert estimate.head_rate_of_spread_m_per_s == pytest.approx(0.29, rel=1e-6)
    assert estimate.back_rate_of_spread_m_per_s == pytest.approx(0.09, rel=1e-6)


def test_the_closure_identity_holds_on_a_clean_series() -> None:
    estimate = estimate_rate_of_spread(
        build_series(0.0, 0.19, 0.10), 0.0, 5, BOOTSTRAP_COUNT, 0
    )
    assert estimate.closure_residual_m_per_s == pytest.approx(0.0, abs=1e-9)
    assert estimate.closure_residual_fraction == pytest.approx(0.0, abs=1e-9)


def test_a_reversed_bearing_leaves_the_closure_identity_satisfied() -> None:
    """The plan says this check catches a wrong bearing. It cannot.

    Reversing the bearing flips the centroid speed and leaves the expansion
    rate alone, so head and back both flip and swap — and their difference is
    still twice the expansion. The rates come back negated and the residual
    stays at zero, which is exactly the failure a consistency check is supposed
    to expose and this one does not.
    """
    series = build_series(0.0, 0.19, 0.10)
    reversed_bearing = estimate_rate_of_spread(series, np.pi, 5, BOOTSTRAP_COUNT, 0)
    assert reversed_bearing.head_rate_of_spread_m_per_s == pytest.approx(
        -0.09, rel=1e-6
    )
    assert reversed_bearing.closure_residual_fraction < 1e-6


def test_a_mis_scaled_semi_axis_also_leaves_the_closure_satisfied() -> None:
    """The plan's other claim for this check, and it fails the same way.

    Scaling every semi-axis by a constant scales the expansion rate by it too,
    so the identity absorbs the factor entirely. A shape factor wrong by half
    would pass unnoticed.
    """
    series = build_series(0.0, 0.19, 0.10)
    mis_scaled = [
        build_extent(
            extent.time_s, extent.centroid_xy_m, 0.5 * extent.semi_axis_major_m
        )
        for extent in series
    ]
    estimate = estimate_rate_of_spread(mis_scaled, 0.0, 5, BOOTSTRAP_COUNT, 0)
    assert estimate.expansion_rate_m_per_s == pytest.approx(0.05, rel=1e-6)
    assert estimate.closure_residual_fraction < 1e-6


def test_a_curving_semi_axis_still_leaves_the_closure_satisfied() -> None:
    """Theil-Sen distributes here, which narrows the check further.

    A median of pairwise slopes does not generally distribute over a sum, but
    it does when one addend has constant pairwise slopes. The centroid is that
    addend, so a straight centroid track makes the identity exact however the
    semi-axis curves.
    """
    direction = np.array([1.0, 0.0])
    times_s = np.arange(1, 11, dtype=np.float64) * 5.0
    curving_semi_axis = [
        build_extent(
            float(time_s),
            CENTRE_XY_M + 0.19 * float(time_s) * direction,
            1.0 + 0.002 * float(time_s) ** 2,
        )
        for time_s in times_s
    ]
    estimate = estimate_rate_of_spread(curving_semi_axis, 0.0, 5, BOOTSTRAP_COUNT, 0)
    assert estimate.closure_residual_fraction < 1e-9


def test_a_curving_centroid_alone_still_leaves_the_closure_satisfied() -> None:
    """The mirror of the previous test, and together they give the real rule.

    Theil-Sen distributes whenever *either* addend has constant pairwise
    slopes. A linear semi-axis is enough on its own, just as a linear centroid
    was, so only a run in which both curve can leave a residual at all.
    """
    direction = np.array([1.0, 0.0])
    times_s = np.arange(1, 11, dtype=np.float64) * 5.0
    curving_centroid = [
        build_extent(
            float(time_s),
            CENTRE_XY_M + 0.004 * float(time_s) ** 2 * direction,
            1.0 + 0.10 * float(time_s),
        )
        for time_s in times_s
    ]
    estimate = estimate_rate_of_spread(curving_centroid, 0.0, 5, BOOTSTRAP_COUNT, 0)
    assert estimate.closure_residual_fraction < 1e-9


def test_even_two_curving_series_leave_the_closure_satisfied() -> None:
    """The third narrowing, and the one that empties the check out.

    Both series quadratic in time makes each pairwise slope a monotone function
    of `t_i + t_j`, so both medians are attained at the same pair and the
    identity survives. Smoothly curving inputs are not enough to disturb it.
    """
    direction = np.array([1.0, 0.0])
    times_s = np.arange(1, 11, dtype=np.float64) * 5.0
    both_curving = [
        build_extent(
            float(time_s),
            CENTRE_XY_M + 0.004 * float(time_s) ** 2 * direction,
            1.0 + 0.002 * float(time_s) ** 2,
        )
        for time_s in times_s
    ]
    estimate = estimate_rate_of_spread(both_curving, 0.0, 5, BOOTSTRAP_COUNT, 0)
    assert estimate.closure_residual_fraction < 1e-9


def test_only_irregular_series_leave_a_residual_at_all() -> None:
    """What is actually left of the plan's free consistency check.

    The residual departs from zero only when the centroid and semi-axis series
    disagree about which pair carries the median slope, which needs noise
    rather than curvature. On the fire scene it reads about two per cent, and
    that is what it is measuring: sampling irregularity in the two series, not
    any property of the physics.
    """
    generator = np.random.default_rng(0)
    direction = np.array([1.0, 0.0])
    times_s = np.arange(1, 11, dtype=np.float64) * 5.0
    noisy = [
        build_extent(
            float(time_s),
            CENTRE_XY_M
            + (0.19 * float(time_s) + generator.normal(0.0, 0.5)) * direction,
            1.0 + 0.10 * float(time_s) + generator.normal(0.0, 0.3),
        )
        for time_s in times_s
    ]
    noisy_estimate = estimate_rate_of_spread(noisy, 0.0, 5, BOOTSTRAP_COUNT, 0)
    clean_estimate = estimate_rate_of_spread(
        build_series(0.0, 0.19, 0.10), 0.0, 5, BOOTSTRAP_COUNT, 0
    )
    assert noisy_estimate.closure_residual_fraction > 1e-6
    assert clean_estimate.closure_residual_fraction < 1e-9


def test_a_front_that_only_expands_has_no_centroid_speed() -> None:
    estimate = estimate_rate_of_spread(
        build_series(0.0, 0.0, 0.12), 0.0, 5, BOOTSTRAP_COUNT, 0
    )
    assert estimate.centroid_speed_m_per_s == pytest.approx(0.0, abs=1e-9)
    assert estimate.head_rate_of_spread_m_per_s == pytest.approx(
        -estimate.back_rate_of_spread_m_per_s, rel=1e-6
    )


def test_a_front_that_only_translates_has_no_expansion() -> None:
    estimate = estimate_rate_of_spread(
        build_series(0.0, 0.2, 0.0), 0.0, 5, BOOTSTRAP_COUNT, 0
    )
    assert estimate.expansion_rate_m_per_s == pytest.approx(0.0, abs=1e-9)
    assert estimate.head_rate_of_spread_m_per_s == pytest.approx(
        estimate.back_rate_of_spread_m_per_s, rel=1e-6
    )


def test_one_bad_early_frame_does_not_set_the_rate() -> None:
    """Why the slopes are Theil-Sen rather than least squares."""
    series = build_series(0.0, 0.19, 0.10)
    corrupted = list(series)
    corrupted[0] = build_extent(5.0, CENTRE_XY_M + np.array([40.0, 0.0]), 30.0)
    robust = estimate_rate_of_spread(corrupted, 0.0, 5, BOOTSTRAP_COUNT, 0)
    clean = estimate_rate_of_spread(series, 0.0, 5, BOOTSTRAP_COUNT, 0)
    assert robust.head_rate_of_spread_m_per_s == pytest.approx(
        clean.head_rate_of_spread_m_per_s, rel=0.05
    )


def test_unresolved_frames_are_left_out_of_the_fit() -> None:
    series = build_series(0.0, 0.19, 0.10)
    with_noise = [
        build_extent(2.5, CENTRE_XY_M + np.array([99.0, 0.0]), 60.0, is_resolved=False),
        *series,
    ]
    estimate = estimate_rate_of_spread(with_noise, 0.0, 5, BOOTSTRAP_COUNT, 0)
    assert estimate.frames_used == len(series)
    assert estimate.head_rate_of_spread_m_per_s == pytest.approx(0.29, rel=1e-6)


def test_too_few_resolved_frames_are_refused() -> None:
    series = build_series(0.0, 0.19, 0.10, frame_count=3)
    with pytest.raises(ValueError, match="at least 5 resolved frames"):
        estimate_rate_of_spread(series, 0.0, 5, BOOTSTRAP_COUNT, 0)


def test_the_interval_brackets_the_rate_it_came_from() -> None:
    estimate = estimate_rate_of_spread(
        build_series(0.0, 0.19, 0.10), 0.0, 5, BOOTSTRAP_COUNT, 0
    )
    lower, upper = estimate.confidence_interval_m_per_s
    assert lower <= estimate.head_rate_of_spread_m_per_s <= upper


def test_the_projection_puts_the_head_ahead_of_the_centroid() -> None:
    series = build_series(0.0, 0.19, 0.10)
    times_s, centroid_m, head_m, back_m, semi_axis_m = project_head_and_back_m(
        series, 0.0
    )
    assert np.all(head_m > centroid_m)
    assert np.all(back_m < centroid_m)
    assert head_m - back_m == pytest.approx(2.0 * semi_axis_m)
    assert times_s.size == len(series)


def test_the_record_serialises_every_field() -> None:
    record = estimate_rate_of_spread(
        build_series(0.0, 0.19, 0.10), 0.0, 5, BOOTSTRAP_COUNT, 0
    ).to_dict()
    assert set(record) == {
        "head_rate_of_spread_m_per_s",
        "back_rate_of_spread_m_per_s",
        "centroid_speed_m_per_s",
        "expansion_rate_m_per_s",
        "bearing_rad",
        "confidence_interval_m_per_s",
        "frames_used",
        "closure_residual_m_per_s",
        "closure_residual_fraction",
    }
