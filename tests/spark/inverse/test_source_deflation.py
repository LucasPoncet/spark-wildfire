from dataclasses import replace

import numpy as np
import pytest

from src.config.multi_source_localization_configuration import (
    NOTCH_DEFLATION,
    SUBSPACE_PROJECTION_DEFLATION,
)
from src.spark.inverse.multilateration import compute_predicted_time_differences_s
from src.spark.inverse.receiver_pair_index import enumerate_receiver_pairs
from src.spark.inverse.source_deflation import (
    apply_subspace_projection_deflation,
    apply_tdoa_notch_deflation,
    build_source_template,
    compute_delay_and_sum_beamformed_signal,
    compute_residual_correlation_energy_ratio,
    deflate_located_source,
)
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
)

SPEED_OF_SOUND_M_PER_S: float = 340.0
RECEIVER_HEIGHT_M: float = 1.5
SOURCE_HEIGHT_M: float = 0.5
NOTCH_WIDTH_S: float = 1e-4
MAXIMUM_ABSOLUTE_LAG_S: float = 0.35


def build_ring_receivers_xyz_m(receiver_count: int = 4) -> np.ndarray:
    bearings_rad = np.linspace(0.0, 2.0 * np.pi, receiver_count, endpoint=False)
    return np.stack(
        (
            50.0 + 45.0 * np.cos(bearings_rad),
            50.0 + 45.0 * np.sin(bearings_rad),
            np.full(receiver_count, RECEIVER_HEIGHT_M),
        ),
        axis=1,
    )


def build_curves_for_sources(
    source_positions_xyz_m: list[np.ndarray],
    source_amplitudes: list[float],
    receivers_xyz_m: np.ndarray,
    receiver_pairs: np.ndarray,
    sample_rate_hz: int,
) -> list[GeneralizedCrossCorrelationCurve]:
    maximum_shift = round(MAXIMUM_ABSOLUTE_LAG_S * sample_rate_hz)
    lags_s = np.arange(-maximum_shift, maximum_shift + 1, dtype=np.float64) / (
        sample_rate_hz
    )
    values = np.zeros((receiver_pairs.shape[0], lags_s.size), dtype=np.float64)
    for position_xyz_m, amplitude in zip(
        source_positions_xyz_m, source_amplitudes, strict=True
    ):
        delays_s = compute_predicted_time_differences_s(
            position_xyz_m, receivers_xyz_m, receiver_pairs, SPEED_OF_SOUND_M_PER_S
        )
        for pair_index, delay_s in enumerate(delays_s):
            values[pair_index] += amplitude * build_source_template(
                lags_s, float(delay_s), NOTCH_WIDTH_S
            )
    return [
        GeneralizedCrossCorrelationCurve(
            receiver_pair=(int(pair[0]), int(pair[1])),
            lags_s=lags_s,
            values=values[pair_index],
            sample_rate_hz=sample_rate_hz,
            window_count=1,
            effective_bandwidth_hz=5000.0,
        )
        for pair_index, pair in enumerate(receiver_pairs)
    ]


def peak_value_at_source(
    curves: list[GeneralizedCrossCorrelationCurve],
    source_position_xyz_m: np.ndarray,
    receivers_xyz_m: np.ndarray,
    receiver_pairs: np.ndarray,
) -> float:
    """The weakest reading a source leaves across the pairs after deflation.

    The minimum rather than the maximum: a source is only still visible to the
    map if every pair still carries it, because the pairwise maps are combined
    by a product.
    """
    delays_s = compute_predicted_time_differences_s(
        source_position_xyz_m, receivers_xyz_m, receiver_pairs, SPEED_OF_SOUND_M_PER_S
    )
    return min(
        float(
            np.abs(curve.values[int(np.argmin(np.abs(curve.lags_s - float(delay_s))))])
        )
        for curve, delay_s in zip(curves, delays_s, strict=True)
    )


def test_the_template_peaks_at_the_predicted_delay() -> None:
    lags_s = np.linspace(-0.01, 0.01, 2001)
    template = build_source_template(lags_s, 0.003, NOTCH_WIDTH_S)
    assert template.max() == pytest.approx(1.0, abs=1e-6)
    assert lags_s[int(np.argmax(template))] == pytest.approx(0.003, abs=1e-5)


def test_notch_deflation_removes_the_peak_of_the_located_source(
    sample_rate_hz: int,
) -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    source_xyz_m = np.array([30.0, 40.0, SOURCE_HEIGHT_M])
    curves = build_curves_for_sources(
        [source_xyz_m], [1.0], receivers_xyz_m, receiver_pairs, sample_rate_hz
    )
    deflated = apply_tdoa_notch_deflation(
        curves,
        source_xyz_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
        NOTCH_WIDTH_S,
    )
    assert (
        peak_value_at_source(deflated, source_xyz_m, receivers_xyz_m, receiver_pairs)
        < 0.01
    )


def test_subspace_deflation_removes_the_peak_of_the_located_source(
    sample_rate_hz: int,
) -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    source_xyz_m = np.array([30.0, 40.0, SOURCE_HEIGHT_M])
    curves = build_curves_for_sources(
        [source_xyz_m], [1.0], receivers_xyz_m, receiver_pairs, sample_rate_hz
    )
    deflated = apply_subspace_projection_deflation(
        curves,
        source_xyz_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
        NOTCH_WIDTH_S,
    )
    assert (
        peak_value_at_source(deflated, source_xyz_m, receivers_xyz_m, receiver_pairs)
        < 0.01
    )


def test_deflating_twice_changes_nothing_further(sample_rate_hz: int) -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    located_xyz_m = np.array([30.0, 40.0, SOURCE_HEIGHT_M])
    other_xyz_m = np.array([70.0, 65.0, SOURCE_HEIGHT_M])
    curves = build_curves_for_sources(
        [located_xyz_m, other_xyz_m],
        [1.0, 0.5],
        receivers_xyz_m,
        receiver_pairs,
        sample_rate_hz,
    )
    once = apply_subspace_projection_deflation(
        curves,
        located_xyz_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
        NOTCH_WIDTH_S,
    )
    twice = apply_subspace_projection_deflation(
        once,
        located_xyz_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
        NOTCH_WIDTH_S,
    )
    assert compute_residual_correlation_energy_ratio(curves, once) < 0.9
    assert compute_residual_correlation_energy_ratio(once, twice) == pytest.approx(
        1.0, abs=1e-9
    )


def test_the_subspace_variant_spares_what_sits_under_the_located_source(
    sample_rate_hz: int,
) -> None:
    """The documented failure mode of this whole family, made measurable.

    A source ten decibels below its neighbour is the one that goes missing when
    the neighbour is peeled off. The notch multiplies the curve down over its
    whole width and takes everything else under it with the located source; the
    projection subtracts only as much as the located source explains, so
    whatever else lives at that lag survives. It over-subtracts somewhat, since
    a background with a non-zero mean inflates the least-squares coefficient, so
    what survives is a fraction of the background rather than all of it.
    """
    receivers_xyz_m = build_ring_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    loud_xyz_m = np.array([30.0, 40.0, SOURCE_HEIGHT_M])
    weak_level = 10.0 ** (-10.0 / 20.0)
    curves = [
        replace(curve, values=curve.values + weak_level)
        for curve in build_curves_for_sources(
            [loud_xyz_m], [1.0], receivers_xyz_m, receiver_pairs, sample_rate_hz
        )
    ]
    notched = apply_tdoa_notch_deflation(
        curves,
        loud_xyz_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
        NOTCH_WIDTH_S,
    )
    projected = apply_subspace_projection_deflation(
        curves,
        loud_xyz_m,
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
        NOTCH_WIDTH_S,
    )
    surviving_under_notch = peak_value_at_source(
        notched, loud_xyz_m, receivers_xyz_m, receiver_pairs
    )
    surviving_under_projection = peak_value_at_source(
        projected, loud_xyz_m, receivers_xyz_m, receiver_pairs
    )
    assert surviving_under_notch < 0.05 * weak_level
    assert surviving_under_projection > 0.3 * weak_level
    assert surviving_under_projection > 10.0 * surviving_under_notch


def test_deflation_never_adds_correlation_energy(sample_rate_hz: int) -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    source_xyz_m = np.array([30.0, 40.0, SOURCE_HEIGHT_M])
    curves = build_curves_for_sources(
        [source_xyz_m], [1.0], receivers_xyz_m, receiver_pairs, sample_rate_hz
    )
    for method in (NOTCH_DEFLATION, SUBSPACE_PROJECTION_DEFLATION):
        deflated = deflate_located_source(
            curves,
            source_xyz_m,
            receivers_xyz_m,
            receiver_pairs,
            SPEED_OF_SOUND_M_PER_S,
            method,
            NOTCH_WIDTH_S,
        )
        assert compute_residual_correlation_energy_ratio(curves, deflated) < 1.0


def test_an_unknown_deflation_method_is_rejected(sample_rate_hz: int) -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m()
    receiver_pairs = enumerate_receiver_pairs(receivers_xyz_m.shape[0])
    curves = build_curves_for_sources(
        [np.array([30.0, 40.0, SOURCE_HEIGHT_M])],
        [1.0],
        receivers_xyz_m,
        receiver_pairs,
        sample_rate_hz,
    )
    with pytest.raises(ValueError, match="unknown deflation method"):
        deflate_located_source(
            curves,
            np.array([30.0, 40.0, SOURCE_HEIGHT_M]),
            receivers_xyz_m,
            receiver_pairs,
            SPEED_OF_SOUND_M_PER_S,
            "nonsense",
            NOTCH_WIDTH_S,
        )


def test_the_beamformer_aligns_the_channels_it_sums(sample_rate_hz: int) -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m()
    source_xyz_m = np.array([30.0, 40.0, SOURCE_HEIGHT_M])
    distances_m = np.linalg.norm(receivers_xyz_m - source_xyz_m, axis=1)
    source_signal = np.random.default_rng(0).standard_normal(sample_rate_hz)

    channels = np.zeros((receivers_xyz_m.shape[0], 2 * sample_rate_hz))
    for receiver_index, distance_m in enumerate(distances_m):
        offset = round(distance_m / SPEED_OF_SOUND_M_PER_S * sample_rate_hz)
        channels[receiver_index, offset : offset + source_signal.size] = source_signal

    beamformed = compute_delay_and_sum_beamformed_signal(
        channels,
        receivers_xyz_m,
        source_xyz_m,
        sample_rate_hz,
        SPEED_OF_SOUND_M_PER_S,
    )
    assert float(np.max(np.abs(beamformed))) > 0.9 * float(
        np.max(np.abs(source_signal))
    )
