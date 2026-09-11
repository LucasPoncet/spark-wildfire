"""Scoring a rate and an extent, and checking the unresolved flag is honest.

The flag check is the one worth having. A stage that reported every frame
unresolved would be perfectly safe and perfectly useless, so what is asserted
is the stronger claim: a frame called unresolved really did have a front the
array could not have resolved.
"""

import numpy as np
import pytest

from src.utils.metrics.spread_metrics import (
    compute_extent_error_series,
    compute_rate_of_spread_error,
    compute_true_rate_of_spread_m_per_s,
    is_flag_honest,
)


def test_a_linear_track_gives_back_its_own_rate() -> None:
    times_s = np.arange(5.0, 55.0, 5.0)
    assert compute_true_rate_of_spread_m_per_s(
        times_s, 0.25 * times_s
    ) == pytest.approx(0.25, rel=1e-9)


def test_one_outlier_does_not_set_the_true_rate() -> None:
    times_s = np.arange(5.0, 55.0, 5.0)
    distances_m = 0.25 * times_s
    corrupted_m = distances_m.copy()
    corrupted_m[0] = 40.0
    assert compute_true_rate_of_spread_m_per_s(times_s, corrupted_m) == pytest.approx(
        0.25, rel=0.05
    )


def test_a_single_observation_has_no_rate() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        compute_true_rate_of_spread_m_per_s(np.array([5.0]), np.array([1.0]))


def test_a_rate_error_reports_both_forms() -> None:
    record = compute_rate_of_spread_error(0.297, 0.250)
    assert record["absolute_error_m_per_s"] == pytest.approx(0.047)
    assert record["relative_error"] == pytest.approx(0.188, abs=0.001)


def test_a_zero_truth_has_no_relative_error_rather_than_an_infinite_one() -> None:
    assert compute_rate_of_spread_error(0.1, 0.0)["relative_error"] is None


def test_extent_errors_are_relative_to_the_truth() -> None:
    errors = compute_extent_error_series(np.array([4.30, 3.60]), np.array([5.33, 4.54]))
    assert errors == pytest.approx([0.1933, 0.2070], abs=0.001)


def test_mismatched_extent_series_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one true semi-axis"):
        compute_extent_error_series(np.zeros(3), np.zeros(2))


def test_a_zero_true_semi_axis_gives_a_missing_error_not_an_infinite_one() -> None:
    errors = compute_extent_error_series(np.array([1.0]), np.array([0.0]))
    assert np.isnan(errors[0])


def test_a_resolved_frame_is_always_honest() -> None:
    assert is_flag_honest(True, 5.0, 1.5)


def test_an_unresolved_frame_below_the_response_is_honest() -> None:
    assert is_flag_honest(False, 1.0, 1.5)


def test_an_unresolved_frame_above_the_response_is_merely_conservative() -> None:
    assert not is_flag_honest(False, 5.0, 1.5)


def test_a_front_exactly_at_the_response_counts_as_honest() -> None:
    assert is_flag_honest(False, 1.5, 1.5)
