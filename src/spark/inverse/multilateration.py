"""Refining a seeded position against every pair delay at once.

The N-receiver analogue of `level_ratio_triangulation.py`. It is seeded by the
steered response power map rather than run on its own: two-step delay-then-solve
methods discard the rest of the correlation function and are fragile under
noise, which is the whole reason the map comes first.
"""

from dataclasses import dataclass

import numpy as np

from src.config.multi_source_localization_configuration import (
    PositionRefinementConfiguration,
)
from src.utils.array_types import Float64Array, Int64Array

DISTANCE_FLOOR_M: float = 1e-9
UNKNOWN_COUNT: int = 2
MINIMUM_RECEIVERS_FOR_INFORMATIVE_RESIDUAL: int = 4


@dataclass(frozen=True)
class PositionEstimate:
    """One source position with everything needed to judge it.

    Attributes:
        position_xy_m: Estimated position `(x, y)` in metres.
        position_covariance_m2: Two-by-two position covariance in metres squared.
        residual_delays_s: Measured minus predicted delay, per pair.
        reduced_chi_square: About one when the delays agree within their stated
            variances. Uninformative below five receivers, where the fit has no
            spare degrees of freedom.
        degrees_of_freedom: Independent delays minus unknowns, that is `N - 3`.
        iteration_count: Gauss-Newton iterations taken.
        has_converged: Whether the step fell below the tolerance.
        is_ambiguous: Whether the geometry admits a mirror solution, which it
            does at exactly three receivers.
    """

    position_xy_m: Float64Array
    position_covariance_m2: Float64Array
    residual_delays_s: Float64Array
    reduced_chi_square: float
    degrees_of_freedom: int
    iteration_count: int
    has_converged: bool
    is_ambiguous: bool


def compute_predicted_time_differences_s(
    position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
) -> Float64Array:
    """Predicts the delay a source at one position puts on every pair.

    Args:
        position_xyz_m: Source position, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.

    Returns:
        Delays of shape `(n_pairs,)`, positive where the first receiver of the
        pair is the farther one.
    """
    position = np.asarray(position_xyz_m, dtype=np.float64)
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    distances_m = np.linalg.norm(receivers - position, axis=1)
    return (
        distances_m[pairs[:, 0]] - distances_m[pairs[:, 1]]
    ) / speed_of_sound_m_per_s


def compute_time_difference_jacobian(
    position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
) -> Float64Array:
    """Differentiates the predicted delays with respect to `(x, y)`.

    Height is held fixed: a ground fire sits at a known height and solving for it
    would trade against range without any observable to pin it.

    Args:
        position_xyz_m: Source position, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.

    Returns:
        Jacobian of shape `(n_pairs, 2)`, in seconds per metre.
    """
    position = np.asarray(position_xyz_m, dtype=np.float64)
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    offsets_m = position - receivers
    distances_m = np.maximum(np.linalg.norm(offsets_m, axis=1), DISTANCE_FLOOR_M)
    unit_directions = offsets_m[:, :2] / distances_m[:, None]
    return (
        unit_directions[pairs[:, 0]] - unit_directions[pairs[:, 1]]
    ) / speed_of_sound_m_per_s


def compute_degrees_of_freedom(receiver_count: int) -> int:
    """Counts the spare degrees of freedom of a delay-only position fit.

    The cocycle constraint means `N` receivers supply only `N - 1` independent
    delays however many pairs are formed, and two of those go on the unknown
    position. At three receivers the fit is exact by construction and its
    residual says nothing; five is where the residual starts to discriminate.

    Args:
        receiver_count: Number of receivers.

    Returns:
        `N - 3`, floored at zero.
    """
    return max(receiver_count - 1 - UNKNOWN_COUNT, 0)


def refine_position_gauss_newton(
    initial_position_xyz_m: Float64Array,
    measured_delays_s: Float64Array,
    delay_variances_s2: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    configuration: PositionRefinementConfiguration,
) -> PositionEstimate:
    """Fits a position to every pair delay, weighted by inverse delay variance.

    Args:
        initial_position_xyz_m: Seed from the steered response power map.
        measured_delays_s: Delay measured on each pair, shape `(n_pairs,)`.
        delay_variances_s2: Variance of each delay, same shape, strictly positive.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        configuration: Iteration ceiling and convergence tolerance.

    Returns:
        The refined position with its covariance and residual diagnostics.

    Raises:
        ValueError: If the delay arrays disagree in length, or any variance is
            not strictly positive.
    """
    delays_s = np.asarray(measured_delays_s, dtype=np.float64)
    variances_s2 = np.asarray(delay_variances_s2, dtype=np.float64)
    if delays_s.size != variances_s2.size:
        raise ValueError("one variance is required per measured delay")
    if np.any(variances_s2 <= 0.0):
        raise ValueError("delay variances must be strictly positive")

    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    position_xyz_m = np.asarray(initial_position_xyz_m, dtype=np.float64).copy()
    weights = 1.0 / variances_s2
    normal_matrix = np.eye(UNKNOWN_COUNT, dtype=np.float64)
    iteration_count = 0
    has_converged = False

    iteration_limit = configuration.maximum_gauss_newton_iterations
    for iteration in range(1, iteration_limit + 1):
        iteration_count = iteration
        residuals_s = delays_s - compute_predicted_time_differences_s(
            position_xyz_m, receivers, receiver_pairs, speed_of_sound_m_per_s
        )
        jacobian = compute_time_difference_jacobian(
            position_xyz_m, receivers, receiver_pairs, speed_of_sound_m_per_s
        )
        normal_matrix = jacobian.T @ (weights[:, None] * jacobian)
        gradient = jacobian.T @ (weights * residuals_s)
        step_m = np.linalg.lstsq(normal_matrix, gradient, rcond=None)[0]
        position_xyz_m[:2] += step_m
        if (
            float(np.linalg.norm(step_m))
            < configuration.position_convergence_tolerance_m
        ):
            has_converged = True
            break

    residuals_s = delays_s - compute_predicted_time_differences_s(
        position_xyz_m, receivers, receiver_pairs, speed_of_sound_m_per_s
    )
    degrees_of_freedom = compute_degrees_of_freedom(receivers.shape[0])
    return PositionEstimate(
        position_xy_m=np.asarray(position_xyz_m[:2], dtype=np.float64),
        position_covariance_m2=np.asarray(
            np.linalg.pinv(normal_matrix), dtype=np.float64
        ),
        residual_delays_s=residuals_s,
        reduced_chi_square=float(
            np.sum(weights * residuals_s**2) / max(degrees_of_freedom, 1)
        ),
        degrees_of_freedom=degrees_of_freedom,
        iteration_count=iteration_count,
        has_converged=has_converged,
        is_ambiguous=receivers.shape[0] < MINIMUM_RECEIVERS_FOR_INFORMATIVE_RESIDUAL,
    )
