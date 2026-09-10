import numpy as np
import pytest

from src.utils.metrics.localization_metrics import (
    build_localization_metric_record,
    compute_optimal_subpattern_assignment_distance,
    compute_position_distance_matrix_m,
    match_estimated_to_true_sources,
)

CUTOFF_M: float = 20.0
ORDER: float = 2.0
TRUE_POSITIONS_XY_M = np.array([[30.0, 40.0], [70.0, 65.0], [50.0, 10.0]])


def test_the_distance_matrix_has_one_row_per_true_source() -> None:
    distances_m = compute_position_distance_matrix_m(
        TRUE_POSITIONS_XY_M, np.array([[30.0, 40.0], [0.0, 0.0]])
    )
    assert distances_m.shape == (3, 2)
    assert distances_m[0, 0] == pytest.approx(0.0)


def test_a_permuted_estimate_set_is_matched_back_to_its_sources() -> None:
    permuted = TRUE_POSITIONS_XY_M[[2, 0, 1]]
    assignment = match_estimated_to_true_sources(TRUE_POSITIONS_XY_M, permuted)
    assert assignment.detection_count == 3
    assert assignment.missed_count == 0
    assert assignment.false_alarm_count == 0
    assert np.allclose(assignment.matched_errors_m, 0.0)
    assert assignment.matched_estimated_indices.tolist() == [1, 2, 0]


def test_a_missing_estimate_is_counted_as_a_miss() -> None:
    assignment = match_estimated_to_true_sources(
        TRUE_POSITIONS_XY_M, TRUE_POSITIONS_XY_M[:2]
    )
    assert assignment.detection_count == 2
    assert assignment.missed_count == 1
    assert assignment.missed_true_indices.tolist() == [2]


def test_a_spurious_estimate_is_counted_as_a_false_alarm() -> None:
    estimates = np.vstack((TRUE_POSITIONS_XY_M, [[5.0, 5.0]]))
    assignment = match_estimated_to_true_sources(TRUE_POSITIONS_XY_M, estimates)
    assert assignment.false_alarm_count == 1
    assert assignment.spurious_estimated_indices.tolist() == [3]


def test_an_empty_estimate_set_misses_everything() -> None:
    assignment = match_estimated_to_true_sources(TRUE_POSITIONS_XY_M, np.empty((0, 2)))
    assert assignment.detection_count == 0
    assert assignment.missed_count == 3


def test_a_perfect_estimate_scores_zero() -> None:
    assert compute_optimal_subpattern_assignment_distance(
        TRUE_POSITIONS_XY_M, TRUE_POSITIONS_XY_M, CUTOFF_M, ORDER
    ) == pytest.approx(0.0)


def test_a_missed_source_is_priced_at_the_cutoff() -> None:
    """The reason this is not a root-mean-square error over matched pairs.

    Finding two of three sources perfectly still costs, because the third was
    never found. An error averaged over matched pairs alone would score it zero.
    """
    distance_m = compute_optimal_subpattern_assignment_distance(
        TRUE_POSITIONS_XY_M, TRUE_POSITIONS_XY_M[:2], CUTOFF_M, ORDER
    )
    assert distance_m == pytest.approx(CUTOFF_M / np.sqrt(3.0))


def test_a_spurious_source_is_priced_the_same_as_a_missed_one() -> None:
    missed_m = compute_optimal_subpattern_assignment_distance(
        TRUE_POSITIONS_XY_M, TRUE_POSITIONS_XY_M[:2], CUTOFF_M, ORDER
    )
    spurious_m = compute_optimal_subpattern_assignment_distance(
        TRUE_POSITIONS_XY_M[:2], TRUE_POSITIONS_XY_M, CUTOFF_M, ORDER
    )
    assert missed_m == pytest.approx(spurious_m)


def test_two_empty_sets_score_zero_and_one_empty_set_scores_the_cutoff() -> None:
    assert compute_optimal_subpattern_assignment_distance(
        np.empty((0, 2)), np.empty((0, 2)), CUTOFF_M, ORDER
    ) == pytest.approx(0.0)
    assert compute_optimal_subpattern_assignment_distance(
        TRUE_POSITIONS_XY_M, np.empty((0, 2)), CUTOFF_M, ORDER
    ) == pytest.approx(CUTOFF_M)


def test_an_error_beyond_the_cutoff_is_clipped_to_it() -> None:
    far_away = TRUE_POSITIONS_XY_M + 1000.0
    assert compute_optimal_subpattern_assignment_distance(
        TRUE_POSITIONS_XY_M, far_away, CUTOFF_M, ORDER
    ) == pytest.approx(CUTOFF_M)


def test_a_non_positive_cutoff_or_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="cutoff_m must be positive"):
        compute_optimal_subpattern_assignment_distance(
            TRUE_POSITIONS_XY_M, TRUE_POSITIONS_XY_M, 0.0, ORDER
        )
    with pytest.raises(ValueError, match="order must be positive"):
        compute_optimal_subpattern_assignment_distance(
            TRUE_POSITIONS_XY_M, TRUE_POSITIONS_XY_M, CUTOFF_M, 0.0
        )


def test_the_metric_record_reports_counts_and_errors() -> None:
    record = build_localization_metric_record(
        TRUE_POSITIONS_XY_M, TRUE_POSITIONS_XY_M[:2], CUTOFF_M, ORDER
    )
    assert record["true_source_count"] == 3
    assert record["estimated_source_count"] == 2
    assert record["detection_count"] == 2
    assert record["missed_count"] == 1
    assert record["false_alarm_count"] == 0
    assert record["mean_error_m"] == pytest.approx(0.0)
