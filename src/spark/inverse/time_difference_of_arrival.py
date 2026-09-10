from dataclasses import dataclass

import numpy as np
from scipy.signal import csd, hilbert, welch

from src.audio.signal_alignment import align_channel_pair
from src.config.multi_source_localization_configuration import (
    EXPONENTIAL_INTERPOLATION,
    PARABOLIC_INTERPOLATION,
    CorrelationConfiguration,
)
from src.config.simulation_configuration import DelayEstimationConfiguration
from src.utils.array_types import BoolArray, ComplexArray, Float64Array, Int64Array

MAXIMUM_COHERENCE: float = 0.999999
SPECTRUM_FLOOR: float = 1e-12


@dataclass(frozen=True)
class TimeDifferenceOfArrival:
    """Delay between two receivers, with the precision of that delay.

    Attributes:
        delay_s: Delay in seconds. Positive means receiver 1 hears the event later.
        variance_s2: Variance of the delay estimate, in seconds squared.
        effective_bandwidth_hz: Cross-spectrum weighted root-mean-square frequency.
        magnitude_squared_coherence: Mean coherence between the aligned channels.
    """

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
    """Estimates the delay between two channels by GCC-PHAT.

    Phase transform weighting whitens the cross-spectrum, which sharpens the
    correlation peak for a broadband source. The peak is refined to sub-sample
    resolution by fitting a parabola to its three highest points.

    Args:
        signal_1: First channel.
        signal_2: Second channel.
        sample_rate_hz: Sample rate in hertz.
        maximum_delay_s: Largest physically possible delay, or None to search all lags.
        interpolation_factor: Spectral zero-padding factor for sub-sample resolution.

    Returns:
        Delay in seconds. Positive means receiver 1 hears the event later.
    """
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
    correlation = np.concatenate(
        (correlation[-maximum_shift:], correlation[: maximum_shift + 1])
    )

    magnitude = np.abs(correlation)
    peak_index = int(np.argmax(magnitude))
    refined_index = float(peak_index)
    if 0 < peak_index < magnitude.size - 1:
        left, centre, right = magnitude[peak_index - 1 : peak_index + 2]
        denominator = left - 2.0 * centre + right
        if denominator != 0.0:
            refined_index += 0.5 * (left - right) / denominator

    return (refined_index - maximum_shift) / float(
        interpolation_factor * sample_rate_hz
    )


def compute_effective_bandwidth_and_coherence(
    signal_1: Float64Array,
    signal_2: Float64Array,
    sample_rate_hz: int,
    configuration: DelayEstimationConfiguration,
) -> tuple[float, float]:
    """Measures effective bandwidth and coherence over the analysis band.

    Both channels must already be aligned. Measured on an unaligned pair, a baseline
    delay comparable to the Welch segment collapses the coherence and the delay
    variance derived from it becomes meaningless.

    Args:
        signal_1: First channel, already aligned.
        signal_2: Second channel, already aligned.
        sample_rate_hz: Sample rate in hertz.
        configuration: Segment length and analysis band limits.

    Returns:
        `(effective_bandwidth_hz, magnitude_squared_coherence)`.
    """
    first = np.asarray(signal_1, dtype=np.float64)
    second = np.asarray(signal_2, dtype=np.float64)
    segment_length = min(
        configuration.coherence_segment_sample_count, first.size, second.size
    )

    frequencies_hz, cross_density = csd(
        first, second, fs=sample_rate_hz, nperseg=segment_length
    )
    _, density_1 = welch(first, fs=sample_rate_hz, nperseg=segment_length)
    _, density_2 = welch(second, fs=sample_rate_hz, nperseg=segment_length)

    coherence = np.abs(cross_density) ** 2 / (density_1 * density_2 + SPECTRUM_FLOOR)
    in_band = (frequencies_hz >= configuration.minimum_analysis_frequency_hz) & (
        frequencies_hz
        <= configuration.maximum_analysis_frequency_fraction_of_nyquist
        * 0.5
        * sample_rate_hz
    )
    if not np.any(in_band):
        in_band = np.ones_like(frequencies_hz, dtype=bool)

    weights = np.abs(cross_density)[in_band]
    weight_sum = float(np.sum(weights)) + SPECTRUM_FLOOR
    effective_bandwidth_hz = float(
        np.sqrt(np.sum(weights * frequencies_hz[in_band] ** 2) / weight_sum)
    )
    mean_coherence = float(
        np.clip(
            np.sum(weights * coherence[in_band]) / weight_sum, 0.0, MAXIMUM_COHERENCE
        )
    )
    return effective_bandwidth_hz, mean_coherence


def compute_delay_variance_s2(
    effective_bandwidth_hz: float,
    magnitude_squared_coherence: float,
    sample_rate_hz: int,
    observation_duration_s: float,
    interpolation_factor: int,
) -> float:
    """Computes the delay variance from bandwidth, coherence and observation time.

    Args:
        effective_bandwidth_hz: Root-mean-square frequency of the cross-spectrum.
        magnitude_squared_coherence: Mean coherence, standing in for the ratio of
            signal power to noise power.
        sample_rate_hz: Sample rate in hertz.
        observation_duration_s: Length of the aligned overlap, in seconds.
        interpolation_factor: Sub-sample resolution factor, setting the floor.

    Returns:
        Variance in seconds squared, floored at the sub-sample resolution.
    """
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
    """Estimates the delay between two receivers and how precise that delay is.

    Args:
        signal_1: First channel.
        signal_2: Second channel.
        sample_rate_hz: Sample rate in hertz.
        maximum_delay_s: Largest physically possible delay, or None to search all lags.
        configuration: Interpolation factor, segment length and analysis band.

    Returns:
        The delay with its variance, effective bandwidth and coherence.
    """
    delay_s = compute_generalised_cross_correlation_phat(
        signal_1,
        signal_2,
        sample_rate_hz,
        maximum_delay_s,
        configuration.interpolation_factor,
    )
    aligned_1, aligned_2 = align_channel_pair(
        signal_1, signal_2, round(delay_s * sample_rate_hz)
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


@dataclass(frozen=True)
class GeneralizedCrossCorrelationCurve:
    """The whole correlation function of one receiver pair, not just its peak.

    Keeping the curve rather than reducing it to a delay is what lets several
    sources be read out of one pair, and what lets a located source be peeled
    out of the correlations without touching the waveforms.

    Attributes:
        receiver_pair: Canonical `(i, j)` indices, `i < j`.
        lags_s: Lag axis in seconds, ascending, shape `(n_lags,)`.
        values: Correlation at each lag, same shape. A positive lag means
            receiver `i` hears the event later.
        sample_rate_hz: Sample rate in hertz.
        window_count: Number of accumulation windows averaged.
        effective_bandwidth_hz: Cross-spectrum weighted root-mean-square
            frequency over the retained band.
    """

    receiver_pair: tuple[int, int]
    lags_s: Float64Array
    values: Float64Array
    sample_rate_hz: int
    window_count: int
    effective_bandwidth_hz: float


@dataclass(frozen=True)
class DelayEstimate:
    """One delay read off a correlation curve.

    Attributes:
        delay_s: Delay in seconds. Positive means receiver `i` hears it later.
        variance_s2: Variance of the delay estimate, in seconds squared.
        peak_value: Height of the correlation peak.
        peak_to_median_ratio: Peak height over the median magnitude of the
            curve, standing in for how prominent the peak is.
    """

    delay_s: float
    variance_s2: float
    peak_value: float
    peak_to_median_ratio: float


def compute_band_mask(
    frequencies_hz: Float64Array,
    lowest_frequency_hz: float,
    highest_frequency_hz: float,
) -> BoolArray:
    """Marks the transform bins the whitening is allowed to touch.

    Args:
        frequencies_hz: Bin centre frequencies, shape `(n_bins,)`.
        lowest_frequency_hz: Lower edge of the retained band, in hertz.
        highest_frequency_hz: Upper edge of that band, in hertz.

    Returns:
        Boolean mask over bins.
    """
    bins_hz = np.asarray(frequencies_hz, dtype=np.float64)
    return np.asarray(
        (bins_hz >= lowest_frequency_hz) & (bins_hz <= highest_frequency_hz),
        dtype=np.bool_,
    )


def compute_whitened_cross_spectrum(
    spectrum_1: ComplexArray,
    spectrum_2: ComplexArray,
    phase_transform_exponent: float,
    phase_transform_regularization: float,
    band_mask: BoolArray,
) -> ComplexArray:
    """Whitens one cross-spectrum inside a band and zeros everything outside it.

    The exponent is the beta of the parameterised phase transform: one is
    conventional PHAT, zero is plain cross-correlation, and values near 0.7
    retain some magnitude information.

    Band-limiting is not cosmetic. Above a lossy encoder brickwall the content
    is the encoder noise floor, and whitening would raise it to full weight in
    the correlation, injecting noise with the same influence as real signal.

    Args:
        spectrum_1: Transform of the first channel.
        spectrum_2: Transform of the second channel.
        phase_transform_exponent: Whitening exponent.
        phase_transform_regularization: Additive term keeping the denominator
            away from zero.
        band_mask: Bins to retain.

    Returns:
        The whitened cross-spectrum, zero outside the band.
    """
    cross_spectrum = np.asarray(spectrum_1) * np.conj(np.asarray(spectrum_2))
    denominator = (
        np.abs(cross_spectrum) ** phase_transform_exponent
        + phase_transform_regularization
    )
    return np.asarray(
        np.where(band_mask, cross_spectrum / denominator, 0.0), dtype=np.complex128
    )


def compute_effective_bandwidth_from_spectrum_hz(
    frequencies_hz: Float64Array,
    cross_spectrum_magnitude: Float64Array,
    band_mask: BoolArray,
) -> float:
    """Computes the root-mean-square frequency of a cross-spectrum in its band.

    Args:
        frequencies_hz: Bin centre frequencies.
        cross_spectrum_magnitude: Magnitude before whitening, used as the weight.
        band_mask: Bins to retain.

    Returns:
        Effective bandwidth in hertz.
    """
    weights = np.asarray(cross_spectrum_magnitude, dtype=np.float64)[band_mask]
    retained_frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)[band_mask]
    weight_sum = float(np.sum(weights)) + SPECTRUM_FLOOR
    return float(np.sqrt(np.sum(weights * retained_frequencies_hz**2) / weight_sum))


def build_correlation_curve_values(
    whitened_cross_spectrum: ComplexArray,
    transform_length: int,
    sample_rate_hz: int,
    maximum_absolute_lag_s: float,
    use_analytic_envelope: bool,
) -> tuple[Float64Array, Float64Array]:
    """Turns a whitened cross-spectrum into a lag-domain curve.

    A band-passed input drives the correlation toward a sinc, whose ripples
    become spurious peaks once the curve is steered over a grid. Taking the
    magnitude of the analytic signal removes them, at the cost of some
    sub-sample sharpness.

    Args:
        whitened_cross_spectrum: Whitened cross-spectrum over the full transform.
        transform_length: Length of the transform the spectrum came from.
        sample_rate_hz: Sample rate in hertz.
        maximum_absolute_lag_s: Largest lag to retain, in seconds.
        use_analytic_envelope: Whether to return the envelope of the curve.

    Returns:
        `(lags_s, values)`, both ascending in lag and of the same length.
    """
    correlation = np.fft.irfft(whitened_cross_spectrum, transform_length)
    centred = np.roll(correlation, transform_length // 2)
    if use_analytic_envelope:
        centred = np.abs(hilbert(centred))
    centre_index = transform_length // 2
    maximum_shift = min(
        round(maximum_absolute_lag_s * sample_rate_hz), centre_index - 1
    )
    values = np.asarray(
        centred[centre_index - maximum_shift : centre_index + maximum_shift + 1],
        dtype=np.float64,
    )
    lags_s = (
        np.arange(-maximum_shift, maximum_shift + 1, dtype=np.float64) / sample_rate_hz
    )
    return lags_s, values


def accumulate_generalized_cross_correlation_over_windows(
    signal_1: Float64Array,
    signal_2: Float64Array,
    receiver_pair: tuple[int, int],
    sample_rate_hz: int,
    maximum_absolute_lag_s: float,
    configuration: CorrelationConfiguration,
) -> GeneralizedCrossCorrelationCurve:
    """Averages the whitened cross-spectrum over windows, then transforms once.

    Fire is continuous noise, so one window gives a correlation peak buried in
    its own variance. Averaging the cross-spectrum over many windows sharpens
    the peak; fifty seconds at half-second windows with half overlap gives about
    two hundred.

    Args:
        signal_1: First channel.
        signal_2: Second channel, same clock and sample rate.
        receiver_pair: Canonical `(i, j)` indices the channels belong to.
        sample_rate_hz: Sample rate in hertz.
        maximum_absolute_lag_s: Largest lag to retain, in seconds.
        configuration: Band, whitening, window and envelope settings.

    Returns:
        The accumulated correlation curve.

    Raises:
        ValueError: If the channels are shorter than one accumulation window.
    """
    first = np.asarray(signal_1, dtype=np.float64)
    second = np.asarray(signal_2, dtype=np.float64)
    sample_count = min(first.size, second.size)
    window_sample_count = round(configuration.window_duration_s * sample_rate_hz)
    if sample_count < window_sample_count:
        raise ValueError("the channels are shorter than one accumulation window")

    hop_sample_count = max(
        round(window_sample_count * (1.0 - configuration.window_overlap)), 1
    )
    transform_length = 1 << int(np.ceil(np.log2(2 * window_sample_count)))
    frequencies_hz = np.fft.rfftfreq(transform_length, d=1.0 / sample_rate_hz)
    band_mask = compute_band_mask(
        frequencies_hz,
        configuration.lowest_frequency_hz,
        configuration.highest_frequency_hz,
    )
    window = np.hanning(window_sample_count)

    accumulated = np.zeros(frequencies_hz.size, dtype=np.complex128)
    accumulated_magnitude = np.zeros(frequencies_hz.size, dtype=np.float64)
    window_count = 0
    for start in range(0, sample_count - window_sample_count + 1, hop_sample_count):
        spectrum_1 = np.fft.rfft(
            first[start : start + window_sample_count] * window, transform_length
        )
        spectrum_2 = np.fft.rfft(
            second[start : start + window_sample_count] * window, transform_length
        )
        accumulated += compute_whitened_cross_spectrum(
            spectrum_1,
            spectrum_2,
            configuration.phase_transform_exponent,
            configuration.phase_transform_regularization,
            band_mask,
        )
        accumulated_magnitude += np.abs(spectrum_1 * np.conj(spectrum_2))
        window_count += 1

    lags_s, values = build_correlation_curve_values(
        accumulated / max(window_count, 1),
        transform_length,
        sample_rate_hz,
        maximum_absolute_lag_s,
        configuration.use_analytic_envelope,
    )
    return GeneralizedCrossCorrelationCurve(
        receiver_pair=receiver_pair,
        lags_s=lags_s,
        values=values,
        sample_rate_hz=sample_rate_hz,
        window_count=window_count,
        effective_bandwidth_hz=compute_effective_bandwidth_from_spectrum_hz(
            frequencies_hz, accumulated_magnitude, band_mask
        ),
    )


def compute_generalized_cross_correlation(
    signal_1: Float64Array,
    signal_2: Float64Array,
    receiver_pair: tuple[int, int],
    sample_rate_hz: int,
    maximum_absolute_lag_s: float,
    configuration: CorrelationConfiguration,
) -> GeneralizedCrossCorrelationCurve:
    """Correlates one pair over the whole record in a single transform.

    Args:
        signal_1: First channel.
        signal_2: Second channel, same clock and sample rate.
        receiver_pair: Canonical `(i, j)` indices the channels belong to.
        sample_rate_hz: Sample rate in hertz.
        maximum_absolute_lag_s: Largest lag to retain, in seconds.
        configuration: Band, whitening and envelope settings.

    Returns:
        The correlation curve, with a window count of one.
    """
    first = np.asarray(signal_1, dtype=np.float64)
    second = np.asarray(signal_2, dtype=np.float64)
    transform_length = 1 << int(np.ceil(np.log2(first.size + second.size)))
    frequencies_hz = np.fft.rfftfreq(transform_length, d=1.0 / sample_rate_hz)
    band_mask = compute_band_mask(
        frequencies_hz,
        configuration.lowest_frequency_hz,
        configuration.highest_frequency_hz,
    )
    spectrum_1 = np.fft.rfft(first, transform_length)
    spectrum_2 = np.fft.rfft(second, transform_length)
    lags_s, values = build_correlation_curve_values(
        compute_whitened_cross_spectrum(
            spectrum_1,
            spectrum_2,
            configuration.phase_transform_exponent,
            configuration.phase_transform_regularization,
            band_mask,
        ),
        transform_length,
        sample_rate_hz,
        maximum_absolute_lag_s,
        configuration.use_analytic_envelope,
    )
    return GeneralizedCrossCorrelationCurve(
        receiver_pair=receiver_pair,
        lags_s=lags_s,
        values=values,
        sample_rate_hz=sample_rate_hz,
        window_count=1,
        effective_bandwidth_hz=compute_effective_bandwidth_from_spectrum_hz(
            frequencies_hz, np.abs(spectrum_1 * np.conj(spectrum_2)), band_mask
        ),
    )


def compute_pair_correlation_curves(
    receiver_signals: Float64Array,
    receiver_pairs: Int64Array,
    maximum_absolute_lag_s: float,
    sample_rate_hz: int,
    configuration: CorrelationConfiguration,
) -> list[GeneralizedCrossCorrelationCurve]:
    """Correlates every receiver pair, accumulating over windows.

    Args:
        receiver_signals: Channels, shape `(n_receivers, n_samples)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        maximum_absolute_lag_s: Largest lag to retain, in seconds.
        sample_rate_hz: Sample rate in hertz.
        configuration: Band, whitening, window and envelope settings.

    Returns:
        One curve per pair, in the order the pairs were given.
    """
    channels = np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
    return [
        accumulate_generalized_cross_correlation_over_windows(
            channels[int(first_index)],
            channels[int(second_index)],
            (int(first_index), int(second_index)),
            sample_rate_hz,
            maximum_absolute_lag_s,
            configuration,
        )
        for first_index, second_index in np.asarray(receiver_pairs, dtype=np.int64)
    ]


def refine_peak_index(
    values: Float64Array, peak_index: int, interpolation: str
) -> float:
    """Refines a peak position to sub-sample resolution.

    Args:
        values: Curve magnitudes.
        peak_index: Index of the largest magnitude.
        interpolation: `parabolic` or `exponential`. Exponential fits a parabola
            to the logarithms, which matches a correlation peak more closely.

    Returns:
        The refined index, fractional.

    Raises:
        ValueError: If the interpolation is not recognised.
    """
    if interpolation not in (PARABOLIC_INTERPOLATION, EXPONENTIAL_INTERPOLATION):
        raise ValueError(f"unknown peak interpolation: {interpolation}")
    if not 0 < peak_index < values.size - 1:
        return float(peak_index)

    neighbourhood = np.asarray(
        values[peak_index - 1 : peak_index + 2], dtype=np.float64
    )
    if interpolation == EXPONENTIAL_INTERPOLATION:
        neighbourhood = np.log(np.maximum(neighbourhood, SPECTRUM_FLOOR))
    left, centre, right = neighbourhood
    denominator = left - 2.0 * centre + right
    if denominator == 0.0:
        return float(peak_index)
    return float(peak_index + 0.5 * (left - right) / denominator)


def estimate_delay_from_correlation_curve(
    curve: GeneralizedCrossCorrelationCurve,
    interpolation: str,
    observation_duration_s: float,
    interpolation_factor: int,
) -> DelayEstimate:
    """Reads the delay off a correlation curve, with its precision.

    The coherence the variance needs is taken from the curve itself: a peak that
    stands far above the rest of the curve came from two well-correlated
    channels, one that barely stands out did not.

    Args:
        curve: The correlation curve.
        interpolation: `parabolic` or `exponential` sub-sample refinement.
        observation_duration_s: Length of the record the curve came from.
        interpolation_factor: Sub-sample resolution factor, setting the variance
            floor.

    Returns:
        The delay with its variance and prominence.
    """
    magnitude = np.abs(curve.values)
    peak_index = int(np.argmax(magnitude))
    refined_index = refine_peak_index(magnitude, peak_index, interpolation)
    delay_s = float(curve.lags_s[0] + refined_index / curve.sample_rate_hz)

    peak_value = float(magnitude[peak_index])
    median_value = float(np.median(magnitude)) + SPECTRUM_FLOOR
    coherence = float(
        np.clip(
            1.0 - median_value / (peak_value + SPECTRUM_FLOOR),
            0.0,
            MAXIMUM_COHERENCE,
        )
    )
    return DelayEstimate(
        delay_s=delay_s,
        variance_s2=compute_delay_variance_s2(
            curve.effective_bandwidth_hz,
            coherence,
            curve.sample_rate_hz,
            observation_duration_s,
            interpolation_factor,
        ),
        peak_value=peak_value,
        peak_to_median_ratio=peak_value / median_value,
    )


def estimate_delay_near_prediction(
    curve: GeneralizedCrossCorrelationCurve,
    predicted_delay_s: float,
    search_half_width_s: float,
    interpolation: str,
    observation_duration_s: float,
    interpolation_factor: int,
) -> DelayEstimate:
    """Reads the delay belonging to one source, not the loudest one on the pair.

    In a mixture the global maximum of a pair curve belongs to whichever source
    dominates that pair, so a source already located by the map has to be read
    out of its own neighbourhood of the curve instead.

    Args:
        curve: The correlation curve.
        predicted_delay_s: Delay the located position predicts for this pair.
        search_half_width_s: How far either side of the prediction to look.
        interpolation: `parabolic` or `exponential` sub-sample refinement.
        observation_duration_s: Length of the record the curve came from.
        interpolation_factor: Sub-sample resolution factor, setting the variance
            floor.

    Returns:
        The delay with its variance and prominence.
    """
    magnitude = np.abs(curve.values)
    inside = np.abs(curve.lags_s - predicted_delay_s) <= search_half_width_s
    if not np.any(inside):
        inside = np.ones_like(magnitude, dtype=np.bool_)
    windowed_magnitude = np.where(inside, magnitude, -np.inf)
    peak_index = int(np.argmax(windowed_magnitude))
    refined_index = refine_peak_index(magnitude, peak_index, interpolation)
    delay_s = float(curve.lags_s[0] + refined_index / curve.sample_rate_hz)

    peak_value = float(magnitude[peak_index])
    median_value = float(np.median(magnitude)) + SPECTRUM_FLOOR
    coherence = float(
        np.clip(
            1.0 - median_value / (peak_value + SPECTRUM_FLOOR),
            0.0,
            MAXIMUM_COHERENCE,
        )
    )
    return DelayEstimate(
        delay_s=delay_s,
        variance_s2=compute_delay_variance_s2(
            curve.effective_bandwidth_hz,
            coherence,
            curve.sample_rate_hz,
            observation_duration_s,
            interpolation_factor,
        ),
        peak_value=peak_value,
        peak_to_median_ratio=peak_value / median_value,
    )
