from dataclasses import dataclass

import numpy as np
from scipy.signal import csd, welch

from src.audio.signal_alignment import align_channel_pair
from src.config.simulation_configuration import DelayEstimationConfiguration
from src.utils.array_types import Float64Array

MAXIMUM_COHERENCE: float = 0.999999
SPECTRUM_FLOOR: float = 1e-12


@dataclass(frozen=True)
class TimeDifferenceOfArrival:
    delay_s: float
    variance_s2: float
    effective_bandwidth_hz: float
    magnitude_squared_coherence: float


def compute_generalised_cross_correlation_phat(
    signal_1: Float64Array,
    signal_2: Float64Array,
    sample_rate_hz: int,
    maximum_delay_s: float | None,
    interpolation_factor: int,
) -> float:
    first = np.asarray(signal_1, dtype=np.float64)
    second = np.asarray(signal_2, dtype=np.float64)
    transform_length = first.size + second.size

    cross_spectrum = np.fft.rfft(first, transform_length) * np.conj(
        np.fft.rfft(second, transform_length)
    )
    cross_spectrum /= np.abs(cross_spectrum) + SPECTRUM_FLOOR
    correlation = np.fft.irfft(cross_spectrum, transform_length * interpolation_factor)

    maximum_shift = int(interpolation_factor * transform_length / 2)
    if maximum_delay_s is not None:
        maximum_shift = min(
            int(interpolation_factor * sample_rate_hz * maximum_delay_s), maximum_shift
        )
    correlation = np.concatenate((correlation[-maximum_shift:], correlation[: maximum_shift + 1]))

    magnitude = np.abs(correlation)
    peak_index = int(np.argmax(magnitude))
    refined_index = float(peak_index)
    if 0 < peak_index < magnitude.size - 1:
        left, centre, right = magnitude[peak_index - 1 : peak_index + 2]
        denominator = left - 2.0 * centre + right
        if denominator != 0.0:
            refined_index += 0.5 * (left - right) / denominator

    return (refined_index - maximum_shift) / float(interpolation_factor * sample_rate_hz)


def compute_effective_bandwidth_and_coherence(
    signal_1: Float64Array,
    signal_2: Float64Array,
    sample_rate_hz: int,
    configuration: DelayEstimationConfiguration,
) -> tuple[float, float]:
    first = np.asarray(signal_1, dtype=np.float64)
    second = np.asarray(signal_2, dtype=np.float64)
    segment_length = min(
        configuration.coherence_segment_sample_count, first.size, second.size
    )

    frequencies_hz, cross_density = csd(first, second, fs=sample_rate_hz, nperseg=segment_length)
    _, density_1 = welch(first, fs=sample_rate_hz, nperseg=segment_length)
    _, density_2 = welch(second, fs=sample_rate_hz, nperseg=segment_length)

    coherence = np.abs(cross_density) ** 2 / (density_1 * density_2 + SPECTRUM_FLOOR)
    in_band = (frequencies_hz >= configuration.minimum_analysis_frequency_hz) & (
        frequencies_hz
        <= configuration.maximum_analysis_frequency_fraction_of_nyquist * 0.5 * sample_rate_hz
    )
    if not np.any(in_band):
        in_band = np.ones_like(frequencies_hz, dtype=bool)

    weights = np.abs(cross_density)[in_band]
    weight_sum = float(np.sum(weights)) + SPECTRUM_FLOOR
    effective_bandwidth_hz = float(
        np.sqrt(np.sum(weights * frequencies_hz[in_band] ** 2) / weight_sum)
    )
    mean_coherence = float(
        np.clip(np.sum(weights * coherence[in_band]) / weight_sum, 0.0, MAXIMUM_COHERENCE)
    )
    return effective_bandwidth_hz, mean_coherence


def compute_delay_variance_s2(
    effective_bandwidth_hz: float,
    magnitude_squared_coherence: float,
    sample_rate_hz: int,
    observation_duration_s: float,
    interpolation_factor: int,
) -> float:
    signal_to_noise_ratio = magnitude_squared_coherence / max(
        1.0 - magnitude_squared_coherence, 1.0 - MAXIMUM_COHERENCE
    )
    time_bandwidth_product = max(observation_duration_s * effective_bandwidth_hz, 1.0)
    standard_deviation_s = 1.0 / (
        2.0
        * np.pi
        * max(effective_bandwidth_hz, 1.0)
        * np.sqrt(signal_to_noise_ratio * time_bandwidth_product)
    )
    resolution_floor_s = 1.0 / (interpolation_factor * sample_rate_hz)
    return float(max(standard_deviation_s, resolution_floor_s) ** 2)


def estimate_time_difference_of_arrival(
    signal_1: Float64Array,
    signal_2: Float64Array,
    sample_rate_hz: int,
    maximum_delay_s: float | None,
    configuration: DelayEstimationConfiguration,
) -> TimeDifferenceOfArrival:
    delay_s = compute_generalised_cross_correlation_phat(
        signal_1,
        signal_2,
        sample_rate_hz,
        maximum_delay_s,
        configuration.interpolation_factor,
    )
    aligned_1, aligned_2 = align_channel_pair(
        signal_1, signal_2, int(round(delay_s * sample_rate_hz))
    )
    effective_bandwidth_hz, coherence = compute_effective_bandwidth_and_coherence(
        aligned_1, aligned_2, sample_rate_hz, configuration
    )
    variance_s2 = compute_delay_variance_s2(
        effective_bandwidth_hz,
        coherence,
        sample_rate_hz,
        aligned_1.size / sample_rate_hz,
        configuration.interpolation_factor,
    )
    return TimeDifferenceOfArrival(
        delay_s=delay_s,
        variance_s2=variance_s2,
        effective_bandwidth_hz=effective_bandwidth_hz,
        magnitude_squared_coherence=coherence,
    )
