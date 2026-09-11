"""Wrapping a bearing error, which is the whole job.

Every assertion here exists because the linear version of the same arithmetic
is plausible and wrong: 350 degrees against 10 degrees is a 20 degree error,
not a 340 degree one, and a summary that averages the latter is unusable.
"""

import numpy as np
import pytest

from src.utils.metrics.bearing_metrics import (
    compute_agreement_fraction,
    compute_bearing_error_series,
    compute_circular_error_rad,
    summarize_bearing_errors,
)


def test_an_error_across_the_wrap_is_the_short_way_round() -> None:
    error_deg = np.rad2deg(
        compute_circular_error_rad(np.deg2rad(10.0), np.deg2rad(350.0))
    )
    assert error_deg == pytest.approx(20.0)


def test_the_error_is_signed() -> None:
    assert compute_circular_error_rad(np.deg2rad(350.0), np.deg2rad(10.0)) < 0.0
    assert compute_circular_error_rad(np.deg2rad(10.0), np.deg2rad(350.0)) > 0.0


def test_an_exact_bearing_has_no_error() -> None:
    assert compute_circular_error_rad(1.234, 1.234) == pytest.approx(0.0, abs=1e-12)


def test_a_half_turn_is_the_largest_error_there_is() -> None:
    assert abs(compute_circular_error_rad(np.deg2rad(180.0), 0.0)) == pytest.approx(
        np.pi
    )


def test_a_series_wraps_every_entry() -> None:
    errors_rad = compute_bearing_error_series(
        np.deg2rad([10.0, 350.0, 180.0]), np.deg2rad([350.0, 10.0, 0.0])
    )
    assert np.rad2deg(errors_rad) == pytest.approx([20.0, -20.0, 180.0])


def test_mismatched_series_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one true bearing"):
        compute_bearing_error_series(np.zeros(3), np.zeros(2))


def test_a_summary_reports_the_median_and_the_tail() -> None:
    summary = summarize_bearing_errors(np.deg2rad([1.0, -2.0, 3.0, -40.0]))
    assert summary["frame_count"] == 4
    assert summary["median_absolute_error_deg"] == pytest.approx(2.5)
    assert summary["ninetieth_percentile_absolute_error_deg"] > 3.0


def test_a_summary_counts_the_frames_inside_each_tolerance() -> None:
    summary = summarize_bearing_errors(np.deg2rad([1.0, 9.0, 15.0, 45.0]))
    assert summary["fraction_within_10_deg"] == pytest.approx(0.5)
    assert summary["fraction_within_20_deg"] == pytest.approx(0.75)


def test_an_empty_series_summarises_to_nothing_rather_than_raising() -> None:
    summary = summarize_bearing_errors(np.empty(0))
    assert summary["frame_count"] == 0
    assert summary["median_absolute_error_deg"] is None


def test_two_methods_that_agree_report_full_agreement() -> None:
    assert compute_agreement_fraction(
        np.deg2rad([10.0, 20.0, 30.0]), np.deg2rad([12.0, 19.0, 33.0]), 15.0
    ) == pytest.approx(1.0)


def test_two_methods_that_disagree_report_none() -> None:
    assert compute_agreement_fraction(
        np.deg2rad([10.0, 20.0]), np.deg2rad([190.0, 200.0]), 15.0
    ) == pytest.approx(0.0)


def test_agreement_is_measured_the_short_way_round_too() -> None:
    assert compute_agreement_fraction(
        np.deg2rad([355.0]), np.deg2rad([5.0]), 15.0
    ) == pytest.approx(1.0)
