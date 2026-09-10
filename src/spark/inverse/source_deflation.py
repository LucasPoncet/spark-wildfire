"""Peeling a located source out of the correlations rather than the waveforms.

Deflating in the correlation domain is what keeps this whole subpackage isolated
from the forward model: subtracting an estimated source from the waveforms would
need a propagation operator, while notching or projecting its contribution out of
the cross-correlations needs only receiver positions, an assumed speed of sound
and the correlations themselves.

The originators of the notch approach report that at three sources the noise in
the correlation function becomes prohibitive. That is a real ceiling on
sequential deflation, which is why `windowed_position_clustering.py` exists as an
independent second route to the source count.
"""

from dataclasses import replace

import numpy as np

from src.audio.signal_alignment import shift_signal_by_samples
from src.config.multi_source_localization_configuration import (
    NOTCH_DEFLATION,
    SUBSPACE_PROJECTION_DEFLATION,
)
from src.spark.inverse.multilateration import compute_predicted_time_differences_s
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
)
from src.utils.array_types import Float64Array, Int64Array

ENERGY_FLOOR: float = 1e-30


def build_source_template(
    lags_s: Float64Array, predicted_delay_s: float, notch_width_s: float
) -> Float64Array:
    """Builds the lag-domain footprint one source leaves on one pair curve.

    A Gaussian of the correlation peak width, centred on the delay the located
    source predicts for this pair.

    Args:
        lags_s: Lag axis of the curve, in seconds.
        predicted_delay_s: Delay the source predicts on this pair, in seconds.
        notch_width_s: Half-width of the footprint, in seconds.

    Returns:
        The template, same shape as `lags_s`, peaking at one.
    """
    offsets_s = np.asarray(lags_s, dtype=np.float64) - predicted_delay_s
    return np.asarray(
        np.exp(-0.5 * (offsets_s / max(notch_width_s, ENERGY_FLOOR)) ** 2),
        dtype=np.float64,
    )


def apply_tdoa_notch_deflation(
    correlation_curves: list[GeneralizedCrossCorrelationCurve],
    source_position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    notch_width_s: float,
) -> list[GeneralizedCrossCorrelationCurve]:
    """Notches out the delay each pair would see from a located source.

    Args:
        correlation_curves: One curve per receiver pair.
        source_position_xyz_m: The located source, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        notch_width_s: Half-width of the notch, in seconds.

    Returns:
        The deflated curves, in the order they were given.
    """
    predicted_delays_s = compute_predicted_time_differences_s(
        source_position_xyz_m,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
    )
    return [
        replace(
            curve,
            values=curve.values
            * (
                1.0
                - build_source_template(
                    curve.lags_s, float(predicted_delay_s), notch_width_s
                )
            ),
        )
        for curve, predicted_delay_s in zip(
            correlation_curves, predicted_delays_s, strict=True
        )
    ]


def apply_subspace_projection_deflation(
    correlation_curves: list[GeneralizedCrossCorrelationCurve],
    source_position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    notch_width_s: float,
) -> list[GeneralizedCrossCorrelationCurve]:
    """Projects each curve onto the subspace orthogonal to a located source.

    Where the notch removes the curve outright over its width, this removes only
    as much as the source actually explains, so a weak source sitting beside a
    strong one survives the strong one being peeled off. That is the documented
    failure mode of every method in this family, and the reason to prefer the
    projection.

    Args:
        correlation_curves: One curve per receiver pair.
        source_position_xyz_m: The located source, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        notch_width_s: Width of the source footprint, in seconds.

    Returns:
        The deflated curves, in the order they were given.
    """
    predicted_delays_s = compute_predicted_time_differences_s(
        source_position_xyz_m,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
    )
    deflated: list[GeneralizedCrossCorrelationCurve] = []
    for curve, predicted_delay_s in zip(
        correlation_curves, predicted_delays_s, strict=True
    ):
        template = build_source_template(
            curve.lags_s, float(predicted_delay_s), notch_width_s
        )
        template_energy = float(np.sum(template**2)) + ENERGY_FLOOR
        projection = float(np.sum(curve.values * template)) / template_energy
        deflated.append(
            replace(curve, values=curve.values - max(projection, 0.0) * template)
        )
    return deflated


def deflate_located_source(
    correlation_curves: list[GeneralizedCrossCorrelationCurve],
    source_position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    method: str,
    notch_width_s: float,
) -> list[GeneralizedCrossCorrelationCurve]:
    """Applies the configured deflation method to every pair curve.

    Args:
        correlation_curves: One curve per receiver pair.
        source_position_xyz_m: The located source, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        method: `tdoa_notch` or `subspace_projection`.
        notch_width_s: Width of the source footprint, in seconds.

    Returns:
        The deflated curves.

    Raises:
        ValueError: If the method is not recognised.
    """
    if method == NOTCH_DEFLATION:
        return apply_tdoa_notch_deflation(
            correlation_curves,
            source_position_xyz_m,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            notch_width_s,
        )
    if method == SUBSPACE_PROJECTION_DEFLATION:
        return apply_subspace_projection_deflation(
            correlation_curves,
            source_position_xyz_m,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            notch_width_s,
        )
    raise ValueError(f"unknown deflation method: {method}")


def compute_delay_and_sum_beamformed_signal(
    receiver_signals: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    source_position_xyz_m: Float64Array,
    sample_rate_hz: int,
    speed_of_sound_m_per_s: float,
) -> Float64Array:
    """Steers the array at one position and averages the aligned channels.

    This is no longer part of deflation; it exists only to give the level stage a
    reference waveform for one source, which the matched filter then projects
    each receiver channel onto.

    Args:
        receiver_signals: Channels, shape `(n_receivers, n_samples)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        source_position_xyz_m: Position to steer at, shape `(3,)`.
        sample_rate_hz: Sample rate in hertz.
        speed_of_sound_m_per_s: Assumed speed of sound.

    Returns:
        The beamformed waveform, shape `(n_samples,)`.
    """
    channels = np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    position = np.asarray(source_position_xyz_m, dtype=np.float64)
    distances_m = np.linalg.norm(receivers - position, axis=1)
    reference_distance_m = float(np.min(distances_m))
    shifts_samples = np.round(
        (reference_distance_m - distances_m) / speed_of_sound_m_per_s * sample_rate_hz
    ).astype(np.int64)
    aligned = np.stack(
        [
            shift_signal_by_samples(channel, int(shift_samples))
            for channel, shift_samples in zip(channels, shifts_samples, strict=True)
        ]
    )
    return np.asarray(aligned.mean(axis=0), dtype=np.float64)


def compute_residual_correlation_energy_ratio(
    original_curves: list[GeneralizedCrossCorrelationCurve],
    deflated_curves: list[GeneralizedCrossCorrelationCurve],
) -> float:
    """Reports how much correlation energy survived a deflation round.

    Args:
        original_curves: Curves before deflation.
        deflated_curves: Curves after deflation.

    Returns:
        Residual energy over original energy, in `[0, 1]` for a deflation that
        only removes.
    """
    original_energy = sum(float(np.sum(curve.values**2)) for curve in original_curves)
    residual_energy = sum(float(np.sum(curve.values**2)) for curve in deflated_curves)
    return float(residual_energy / (original_energy + ENERGY_FLOOR))
