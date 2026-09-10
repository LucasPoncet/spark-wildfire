"""Scoring a set of estimated positions against a set of true positions.

Root-mean-square error over matched pairs silently ignores missed and spurious
sources, which is the failure mode a multi-source sweep exists to expose. The
optimal sub-pattern assignment distance does not: it prices a cardinality error
at the cutoff, so a run that finds one source out of three cannot look good by
locating that one precisely.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.array_types import Float64Array, Int64Array


@dataclass(frozen=True)
class SourceAssignment:
    """Which estimate was matched to which true source.

    Attributes:
        matched_true_indices: Index into the true positions, per matched pair.
        matched_estimated_indices: Index into the estimates, per matched pair.
        matched_errors_m: Distance of each matched pair, in metres.
        missed_true_indices: True sources no estimate was matched to.
        spurious_estimated_indices: Estimates matched to no true source.
    """

    matched_true_indices: Int64Array
    matched_estimated_indices: Int64Array
    matched_errors_m: Float64Array
    missed_true_indices: Int64Array
    spurious_estimated_indices: Int64Array

    @property
    def detection_count(self) -> int:
        """Number of true sources that were matched."""
        return int(self.matched_true_indices.size)

    @property
    def missed_count(self) -> int:
        """Number of true sources that were not found."""
        return int(self.missed_true_indices.size)

    @property
    def false_alarm_count(self) -> int:
        """Number of estimates that correspond to no true source."""
        return int(self.spurious_estimated_indices.size)


def compute_position_distance_matrix_m(
    true_positions_xy_m: Float64Array, estimated_positions_xy_m: Float64Array
) -> Float64Array:
    """Measures every true-to-estimated distance.

    Args:
        true_positions_xy_m: True positions, shape `(n_true, 2)`.
        estimated_positions_xy_m: Estimated positions, shape `(n_estimated, 2)`.

    Returns:
        Distances of shape `(n_true, n_estimated)`, in metres.
    """
    true_positions = np.atleast_2d(np.asarray(true_positions_xy_m, dtype=np.float64))
    estimated_positions = np.atleast_2d(
        np.asarray(estimated_positions_xy_m, dtype=np.float64)
    )
    return np.linalg.norm(
        true_positions[:, None, :] - estimated_positions[None, :, :], axis=2
    )


def match_estimated_to_true_sources(
    true_positions_xy_m: Float64Array, estimated_positions_xy_m: Float64Array
) -> SourceAssignment:
    """Pairs estimates with true sources so total distance is smallest.

    Args:
        true_positions_xy_m: True positions, shape `(n_true, 2)`.
        estimated_positions_xy_m: Estimated positions, shape `(n_estimated, 2)`.

    Returns:
        The assignment, with whatever was left unmatched on either side.
    """
    true_positions = np.atleast_2d(np.asarray(true_positions_xy_m, dtype=np.float64))
    estimated_positions = np.atleast_2d(
        np.asarray(estimated_positions_xy_m, dtype=np.float64)
    )
    if true_positions.size == 0 or estimated_positions.size == 0:
        return SourceAssignment(
            matched_true_indices=np.empty(0, dtype=np.int64),
            matched_estimated_indices=np.empty(0, dtype=np.int64),
            matched_errors_m=np.empty(0, dtype=np.float64),
            missed_true_indices=np.arange(true_positions.shape[0], dtype=np.int64),
            spurious_estimated_indices=np.arange(
                estimated_positions.shape[0], dtype=np.int64
            ),
        )

    distances_m = compute_position_distance_matrix_m(
        true_positions, estimated_positions
    )
    true_indices, estimated_indices = linear_sum_assignment(distances_m)
    return SourceAssignment(
        matched_true_indices=np.asarray(true_indices, dtype=np.int64),
        matched_estimated_indices=np.asarray(estimated_indices, dtype=np.int64),
        matched_errors_m=np.asarray(
            distances_m[true_indices, estimated_indices], dtype=np.float64
        ),
        missed_true_indices=np.setdiff1d(
            np.arange(true_positions.shape[0], dtype=np.int64), true_indices
        ),
        spurious_estimated_indices=np.setdiff1d(
            np.arange(estimated_positions.shape[0], dtype=np.int64), estimated_indices
        ),
    )


def compute_optimal_subpattern_assignment_distance(
    true_positions_xy_m: Float64Array,
    estimated_positions_xy_m: Float64Array,
    cutoff_m: float,
    order: float,
) -> float:
    """Scores a whole set of estimates, cardinality errors included.

    Args:
        true_positions_xy_m: True positions, shape `(n_true, 2)`.
        estimated_positions_xy_m: Estimated positions, shape `(n_estimated, 2)`.
        cutoff_m: Distance at which a localization error is as bad as a missed
            or spurious source, in metres.
        order: Exponent of the underlying distance, usually one or two.

    Returns:
        The distance in metres. Zero when the two sets coincide, `cutoff_m` when
        one set is empty and the other is not.

    Raises:
        ValueError: If the cutoff or the order is not positive.
    """
    if cutoff_m <= 0.0:
        raise ValueError("cutoff_m must be positive")
    if order <= 0.0:
        raise ValueError("order must be positive")

    true_positions = np.atleast_2d(np.asarray(true_positions_xy_m, dtype=np.float64))
    estimated_positions = np.atleast_2d(
        np.asarray(estimated_positions_xy_m, dtype=np.float64)
    )
    true_count = 0 if true_positions.size == 0 else true_positions.shape[0]
    estimated_count = (
        0 if estimated_positions.size == 0 else estimated_positions.shape[0]
    )
    if true_count == 0 and estimated_count == 0:
        return 0.0
    if true_count == 0 or estimated_count == 0:
        return float(cutoff_m)

    assignment = match_estimated_to_true_sources(true_positions, estimated_positions)
    clipped_errors_m = np.minimum(assignment.matched_errors_m, cutoff_m)
    larger_count = max(true_count, estimated_count)
    cardinality_penalty = cutoff_m**order * abs(true_count - estimated_count)
    return float(
        ((np.sum(clipped_errors_m**order) + cardinality_penalty) / larger_count)
        ** (1.0 / order)
    )


def build_localization_metric_record(
    true_positions_xy_m: Float64Array,
    estimated_positions_xy_m: Float64Array,
    cutoff_m: float,
    order: float,
) -> dict[str, object]:
    """Summarises one localization result for the metrics document.

    Args:
        true_positions_xy_m: True positions, shape `(n_true, 2)`.
        estimated_positions_xy_m: Estimated positions, shape `(n_estimated, 2)`.
        cutoff_m: Optimal sub-pattern assignment cutoff, in metres.
        order: Optimal sub-pattern assignment order.

    Returns:
        A JSON-serializable record.
    """
    assignment = match_estimated_to_true_sources(
        true_positions_xy_m, estimated_positions_xy_m
    )
    errors_m = assignment.matched_errors_m
    return {
        "true_source_count": int(
            np.atleast_2d(np.asarray(true_positions_xy_m)).shape[0]
        ),
        "estimated_source_count": int(
            np.atleast_2d(np.asarray(estimated_positions_xy_m)).shape[0]
            if np.asarray(estimated_positions_xy_m).size
            else 0
        ),
        "detection_count": assignment.detection_count,
        "missed_count": assignment.missed_count,
        "false_alarm_count": assignment.false_alarm_count,
        "matched_errors_m": errors_m,
        "mean_error_m": float(errors_m.mean()) if errors_m.size else float("nan"),
        "median_error_m": float(np.median(errors_m)) if errors_m.size else float("nan"),
        "maximum_error_m": float(errors_m.max()) if errors_m.size else float("nan"),
        "optimal_subpattern_assignment_m": (
            compute_optimal_subpattern_assignment_distance(
                true_positions_xy_m, estimated_positions_xy_m, cutoff_m, order
            )
        ),
        "optimal_subpattern_assignment_cutoff_m": cutoff_m,
        "optimal_subpattern_assignment_order": order,
    }
