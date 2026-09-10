"""Fusing delays and band levels into one position solve, per source.

This is where the two observables earn their keep together. Delays constrain
range *differences* very precisely but a compact array constrains absolute range
poorly, so a delay-only error ellipse stretches radially away from the array.
The absorption slope across bands supplies exactly that missing direction: it
depends on absolute path length, not on path difference.

Absorption coefficients arrive as an argument and are never recomputed here, so
`atmosphere/` stays the single definition the renderer and the estimator share.
"""

from dataclasses import dataclass

import numpy as np

from src.audio.matched_filter_band_level import (
    compute_matched_filter_band_levels_db,
)
from src.audio.signal_alignment import shift_signal_by_samples
from src.config.multi_source_localization_configuration import (
    PositionRefinementConfiguration,
)
from src.spark.inverse.inverse_variance_fusion import fuse_inverse_variance
from src.spark.inverse.multilateration import (
    compute_predicted_time_differences_s,
    compute_time_difference_jacobian,
)
from src.spark.inverse.source_deflation import (
    compute_delay_and_sum_beamformed_signal,
)
from src.utils.array_types import Float64Array, Int64Array

DISTANCE_FLOOR_M: float = 1e-9
UNKNOWN_COUNT: int = 2
DECIBELS_PER_NATURAL_LOG: float = 20.0 / np.log(10.0)
LEVEL_VARIANCE_FLOOR_DB2: float = 1e-6


@dataclass(frozen=True)
class LevelRefinedPosition:
    """A position fitted to delays and band levels at once.

    Attributes:
        position_xy_m: Estimated position `(x, y)` in metres.
        position_covariance_m2: Two-by-two position covariance in metres squared.
        delay_residuals_s: Measured minus predicted delay, per pair.
        level_residuals_db: Measured minus predicted geometric level difference,
            per pair.
        band_center_frequencies_hz: Bands the levels were measured in.
        pair_geometric_level_difference_db: Fused level difference per pair.
        pair_geometric_level_variance_db2: Variance of each of those.
        reduced_chi_square: About one when delays and levels agree within their
            stated uncertainties.
        iteration_count: Gauss-Newton iterations taken.
        has_converged: Whether the step fell below the tolerance.
    """

    position_xy_m: Float64Array
    position_covariance_m2: Float64Array
    delay_residuals_s: Float64Array
    level_residuals_db: Float64Array
    band_center_frequencies_hz: Float64Array
    pair_geometric_level_difference_db: Float64Array
    pair_geometric_level_variance_db2: Float64Array
    reduced_chi_square: float
    iteration_count: int
    has_converged: bool


def align_channels_to_beamformer(
    receiver_signals: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    source_position_xyz_m: Float64Array,
    sample_rate_hz: int,
    speed_of_sound_m_per_s: float,
) -> Float64Array:
    """Puts every receiver channel on the clock the beamformer steered to.

    The matched filter cancels part of its own projection when a residual delay
    puts a phase ramp across a band, so the channels have to be shifted by the
    same amounts the beamformer used before their levels are read.

    Args:
        receiver_signals: Channels, shape `(n_receivers, n_samples)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        source_position_xyz_m: Position steered at, shape `(3,)`.
        sample_rate_hz: Sample rate in hertz.
        speed_of_sound_m_per_s: Assumed speed of sound.

    Returns:
        The aligned channels, same shape as the input.
    """
    channels = np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    distances_m = np.linalg.norm(
        receivers - np.asarray(source_position_xyz_m, dtype=np.float64), axis=1
    )
    shifts_samples = np.round(
        (float(np.min(distances_m)) - distances_m)
        / speed_of_sound_m_per_s
        * sample_rate_hz
    ).astype(np.int64)
    return np.stack(
        [
            shift_signal_by_samples(channel, int(shift_samples))
            for channel, shift_samples in zip(channels, shifts_samples, strict=True)
        ]
    )


def compute_pair_level_differences_db(
    receiver_band_levels_db: Float64Array,
    receiver_band_level_variances_db2: Float64Array,
    receiver_pairs: Int64Array,
) -> tuple[Float64Array, Float64Array]:
    """Differences the per-receiver band levels across every pair.

    Args:
        receiver_band_levels_db: Levels, shape `(n_receivers, n_bands)`.
        receiver_band_level_variances_db2: Variance of each level, same shape.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.

    Returns:
        `(differences_db, variances_db2)`, both of shape `(n_pairs, n_bands)`.
    """
    levels_db = np.asarray(receiver_band_levels_db, dtype=np.float64)
    variances_db2 = np.asarray(receiver_band_level_variances_db2, dtype=np.float64)
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    return (
        levels_db[pairs[:, 0]] - levels_db[pairs[:, 1]],
        variances_db2[pairs[:, 0]]
        + variances_db2[pairs[:, 1]]
        + LEVEL_VARIANCE_FLOOR_DB2,
    )


def compute_predicted_level_differences_db(
    position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
) -> Float64Array:
    """Predicts the geometric level difference a position puts on every pair.

    Args:
        position_xyz_m: Source position, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.

    Returns:
        Level differences of shape `(n_pairs,)`, in decibels.
    """
    distances_m = np.maximum(
        np.linalg.norm(
            np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
            - np.asarray(position_xyz_m, dtype=np.float64),
            axis=1,
        ),
        DISTANCE_FLOOR_M,
    )
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    return DECIBELS_PER_NATURAL_LOG * np.log(
        distances_m[pairs[:, 1]] / distances_m[pairs[:, 0]]
    )


def compute_level_difference_jacobian(
    position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
) -> Float64Array:
    """Differentiates the predicted level differences with respect to `(x, y)`.

    Args:
        position_xyz_m: Source position, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.

    Returns:
        Jacobian of shape `(n_pairs, 2)`, in decibels per metre.
    """
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    offsets_m = np.asarray(position_xyz_m, dtype=np.float64) - receivers
    distances_m = np.maximum(np.linalg.norm(offsets_m, axis=1), DISTANCE_FLOOR_M)
    scaled_directions = offsets_m[:, :2] / (distances_m**2)[:, None]
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    return np.asarray(
        DECIBELS_PER_NATURAL_LOG
        * (scaled_directions[pairs[:, 1]] - scaled_directions[pairs[:, 0]]),
        dtype=np.float64,
    )


def fuse_pair_geometric_level_differences(
    pair_level_differences_db: Float64Array,
    pair_level_variances_db2: Float64Array,
    absorption_coefficients_db_per_m: Float64Array,
    pair_path_differences_m: Float64Array,
) -> tuple[Float64Array, Float64Array]:
    """Removes absorption from each band and fuses the bands into one term.

    Each band is an independent measurement of the same geometric quantity once
    the absorption its own coefficient accounts for has been subtracted, so the
    bands are fused by inverse variance rather than averaged.

    Args:
        pair_level_differences_db: Measured differences, shape
            `(n_pairs, n_bands)`.
        pair_level_variances_db2: Variance of each, same shape.
        absorption_coefficients_db_per_m: One coefficient per band.
        pair_path_differences_m: Range difference `r_i - r_j` per pair, from the
            current position estimate.

    Returns:
        `(fused_db, fused_variance_db2)`, both of shape `(n_pairs,)`.
    """
    differences_db = np.atleast_2d(
        np.asarray(pair_level_differences_db, dtype=np.float64)
    )
    variances_db2 = np.atleast_2d(
        np.asarray(pair_level_variances_db2, dtype=np.float64)
    )
    absorption = np.asarray(absorption_coefficients_db_per_m, dtype=np.float64)
    path_differences_m = np.asarray(pair_path_differences_m, dtype=np.float64)

    geometric_db = differences_db + absorption[None, :] * path_differences_m[:, None]
    fused = [
        fuse_inverse_variance(pair_values_db, pair_variances_db2)
        for pair_values_db, pair_variances_db2 in zip(
            geometric_db, variances_db2, strict=True
        )
    ]
    return (
        np.array([estimate.value for estimate in fused], dtype=np.float64),
        np.array([estimate.variance for estimate in fused], dtype=np.float64),
    )


def refine_position_with_band_levels(
    initial_position_xyz_m: Float64Array,
    measured_delays_s: Float64Array,
    delay_variances_s2: Float64Array,
    receiver_signals: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    sample_rate_hz: int,
    band_center_frequencies_hz: Float64Array,
    band_edges_hz: Float64Array,
    absorption_coefficients_db_per_m: Float64Array,
    speed_of_sound_m_per_s: float,
    window_sample_count: int,
    window_overlap_fraction: float,
    configuration: PositionRefinementConfiguration,
) -> LevelRefinedPosition:
    """Fits one source position to its pair delays and its band levels together.

    Args:
        initial_position_xyz_m: Seed, normally the delay-only estimate.
        measured_delays_s: Delay attributed to this source per pair.
        delay_variances_s2: Variance of each of those delays.
        receiver_signals: Channels, shape `(n_receivers, n_samples)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        sample_rate_hz: Sample rate in hertz.
        band_center_frequencies_hz: Band centres, shape `(n_bands,)`.
        band_edges_hz: Lower and upper edge per band, shape `(n_bands, 2)`.
        absorption_coefficients_db_per_m: One coefficient per band, handed in.
        speed_of_sound_m_per_s: Assumed speed of sound.
        window_sample_count: Analysis window for the band levels, in samples.
        window_overlap_fraction: Overlap for that window.
        configuration: Iteration ceiling, tolerance and whether to use the level
            terms at all.

    Returns:
        The refined position with its covariance and residual diagnostics.
    """
    channels = np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    position_xyz_m = np.asarray(initial_position_xyz_m, dtype=np.float64).copy()
    delays_s = np.asarray(measured_delays_s, dtype=np.float64)
    delay_weights = 1.0 / np.asarray(delay_variances_s2, dtype=np.float64)
    pairs = np.asarray(receiver_pairs, dtype=np.int64)

    reference_signal = compute_delay_and_sum_beamformed_signal(
        channels, receivers, position_xyz_m, sample_rate_hz, speed_of_sound_m_per_s
    )
    band_levels = compute_matched_filter_band_levels_db(
        reference_signal,
        align_channels_to_beamformer(
            channels, receivers, position_xyz_m, sample_rate_hz, speed_of_sound_m_per_s
        ),
        band_center_frequencies_hz,
        band_edges_hz,
        sample_rate_hz,
        window_sample_count,
        window_overlap_fraction,
    )
    pair_level_differences_db, pair_level_variances_db2 = (
        compute_pair_level_differences_db(
            band_levels.levels_db, band_levels.window_variance_db2, pairs
        )
    )

    normal_matrix = np.eye(UNKNOWN_COUNT, dtype=np.float64)
    fused_level_db = np.zeros(pairs.shape[0], dtype=np.float64)
    fused_level_variance_db2 = np.ones(pairs.shape[0], dtype=np.float64)
    delay_residuals_s = np.zeros(pairs.shape[0], dtype=np.float64)
    level_residuals_db = np.zeros(pairs.shape[0], dtype=np.float64)
    iteration_count = 0
    has_converged = False

    for iteration in range(1, configuration.maximum_gauss_newton_iterations + 1):
        iteration_count = iteration
        delay_residuals_s = delays_s - compute_predicted_time_differences_s(
            position_xyz_m, receivers, pairs, speed_of_sound_m_per_s
        )
        delay_jacobian = compute_time_difference_jacobian(
            position_xyz_m, receivers, pairs, speed_of_sound_m_per_s
        )
        if not configuration.use_band_level_terms:
            residuals = delay_residuals_s
            weights = delay_weights
            jacobian = delay_jacobian
        else:
            distances_m = np.linalg.norm(receivers - position_xyz_m, axis=1)
            fused_level_db, fused_level_variance_db2 = (
                fuse_pair_geometric_level_differences(
                    pair_level_differences_db,
                    pair_level_variances_db2,
                    absorption_coefficients_db_per_m,
                    distances_m[pairs[:, 0]] - distances_m[pairs[:, 1]],
                )
            )
            level_residuals_db = (
                fused_level_db
                - compute_predicted_level_differences_db(
                    position_xyz_m, receivers, pairs
                )
            )
            residuals = np.concatenate((delay_residuals_s, level_residuals_db))
            weights = np.concatenate((delay_weights, 1.0 / fused_level_variance_db2))
            jacobian = np.vstack(
                (
                    delay_jacobian,
                    compute_level_difference_jacobian(position_xyz_m, receivers, pairs),
                )
            )

        normal_matrix = jacobian.T @ (weights[:, None] * jacobian)
        step_m = np.linalg.lstsq(
            normal_matrix, jacobian.T @ (weights * residuals), rcond=None
        )[0]
        position_xyz_m[:2] += step_m
        if (
            float(np.linalg.norm(step_m))
            < configuration.position_convergence_tolerance_m
        ):
            has_converged = True
            break

    residual_count = delay_residuals_s.size + (
        level_residuals_db.size if configuration.use_band_level_terms else 0
    )
    weighted_square_sum = float(np.sum(delay_weights * delay_residuals_s**2))
    if configuration.use_band_level_terms:
        weighted_square_sum += float(
            np.sum(level_residuals_db**2 / fused_level_variance_db2)
        )
    return LevelRefinedPosition(
        position_xy_m=np.asarray(position_xyz_m[:2], dtype=np.float64),
        position_covariance_m2=np.asarray(
            np.linalg.pinv(normal_matrix), dtype=np.float64
        ),
        delay_residuals_s=delay_residuals_s,
        level_residuals_db=level_residuals_db,
        band_center_frequencies_hz=np.asarray(
            band_center_frequencies_hz, dtype=np.float64
        ),
        pair_geometric_level_difference_db=fused_level_db,
        pair_geometric_level_variance_db2=fused_level_variance_db2,
        reduced_chi_square=weighted_square_sum / max(residual_count - UNKNOWN_COUNT, 1),
        iteration_count=iteration_count,
        has_converged=has_converged,
    )
