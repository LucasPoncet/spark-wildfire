import numpy as np
import pytest

from src.audio.signal_alignment import shift_signal_by_samples
from src.config.multi_source_localization_configuration import (
    EXPONENTIAL_INTERPOLATION,
    PARABOLIC_INTERPOLATION,
    CorrelationConfiguration,
)
from src.config.simulation_configuration import DelayEstimationConfiguration
from src.spark.inverse.receiver_pair_index import enumerate_receiver_pairs
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
    accumulate_generalized_cross_correlation_over_windows,
    compute_delay_variance_s2,
    compute_generalised_cross_correlation_phat,
    compute_generalized_cross_correlation,
    compute_pair_correlation_curves,
    estimate_delay_from_correlation_curve,
    estimate_delay_near_prediction,
    estimate_time_difference_of_arrival,
    refine_peak_index,
)

SEARCH_WINDOW_S: float = 0.02


def test_delay_is_positive_when_the_first_receiver_is_further_away(
    sample_rate_hz: int, delay_estimation: DelayEstimationConfiguration
) -> None:
    base = np.random.default_rng(0).normal(size=sample_rate_hz)
    delay_samples = 137
    arrival = estimate_time_difference_of_arrival(
        shift_signal_by_samples(base, delay_samples),
        base,
        sample_rate_hz,
        SEARCH_WINDOW_S,
        delay_estimation,
    )
    assert arrival.delay_s * sample_rate_hz == pytest.approx(delay_samples, abs=0.1)


def test_delay_is_negative_when_the_second_receiver_is_further_away(
    sample_rate_hz: int, delay_estimation: DelayEstimationConfiguration
) -> None:
    base = np.random.default_rng(1).normal(size=sample_rate_hz)
    delay_samples = 91
    arrival = estimate_time_difference_of_arrival(
        base,
        shift_signal_by_samples(base, delay_samples),
        sample_rate_hz,
        SEARCH_WINDOW_S,
        delay_estimation,
    )
    assert arrival.delay_s * sample_rate_hz == pytest.approx(-delay_samples, abs=0.1)


def test_delay_variance_grows_as_the_channels_decorrelate(
    sample_rate_hz: int, delay_estimation: DelayEstimationConfiguration
) -> None:
    generator = np.random.default_rng(2)
    base = generator.normal(size=sample_rate_hz)
    delayed = shift_signal_by_samples(base, 50)
    variances = [
        estimate_time_difference_of_arrival(
            delayed + generator.normal(scale=noise_scale, size=delayed.size),
            base,
            sample_rate_hz,
            SEARCH_WINDOW_S,
            delay_estimation,
        ).variance_s2
        for noise_scale in [0.01, 0.1, 1.0]
    ]
    assert variances[0] <= variances[1] <= variances[2]


def test_maximum_delay_bounds_the_search(
    sample_rate_hz: int, delay_estimation: DelayEstimationConfiguration
) -> None:
    base = np.random.default_rng(3).normal(size=sample_rate_hz)
    arrival = estimate_time_difference_of_arrival(
        shift_signal_by_samples(base, 4000),
        base,
        sample_rate_hz,
        0.01,
        delay_estimation,
    )
    assert abs(arrival.delay_s) <= 0.01


def test_unbounded_search_still_finds_the_delay(
    sample_rate_hz: int, delay_estimation: DelayEstimationConfiguration
) -> None:
    base = np.random.default_rng(4).normal(size=sample_rate_hz)
    delay_samples = 220
    arrival = estimate_time_difference_of_arrival(
        shift_signal_by_samples(base, delay_samples),
        base,
        sample_rate_hz,
        None,
        delay_estimation,
    )
    assert arrival.delay_s * sample_rate_hz == pytest.approx(delay_samples, abs=0.1)


CONVENTIONAL_PHASE_TRANSFORM_EXPONENT: float = 1.0
CORRELATION_BAND_LOW_HZ: float = 250.0
CORRELATION_BAND_HIGH_HZ: float = 11000.0
CURVE_MAXIMUM_LAG_S: float = 0.05


def build_correlation_configuration(
    use_analytic_envelope: bool = False,
    phase_transform_exponent: float = CONVENTIONAL_PHASE_TRANSFORM_EXPONENT,
    highest_frequency_hz: float = CORRELATION_BAND_HIGH_HZ,
    peak_interpolation: str = PARABOLIC_INTERPOLATION,
) -> CorrelationConfiguration:
    return CorrelationConfiguration(
        lowest_frequency_hz=CORRELATION_BAND_LOW_HZ,
        highest_frequency_hz=highest_frequency_hz,
        phase_transform_exponent=phase_transform_exponent,
        phase_transform_regularization=1e-8,
        use_analytic_envelope=use_analytic_envelope,
        window_duration_s=0.25,
        window_overlap=0.5,
        maximum_absolute_lag_margin=1.1,
        peak_interpolation=peak_interpolation,
    )


def make_delayed_pair(
    sample_rate_hz: int, delay_samples: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Channel one hears the event `delay_samples` later than channel two."""
    generator = np.random.default_rng(seed)
    source = generator.standard_normal(2 * sample_rate_hz)
    padding = np.zeros(abs(delay_samples))
    return (
        np.concatenate((padding, source)),
        np.concatenate((source, padding)),
    )


def test_a_positive_lag_means_the_first_receiver_hears_it_later(
    sample_rate_hz: int,
) -> None:
    delay_samples = 400
    signal_1, signal_2 = make_delayed_pair(sample_rate_hz, delay_samples)
    curve = compute_generalized_cross_correlation(
        signal_1,
        signal_2,
        (0, 1),
        sample_rate_hz,
        CURVE_MAXIMUM_LAG_S,
        build_correlation_configuration(),
    )
    estimate = estimate_delay_from_correlation_curve(
        curve, PARABOLIC_INTERPOLATION, 2.0, 16
    )
    assert estimate.delay_s == pytest.approx(delay_samples / sample_rate_hz, abs=1e-5)


def test_the_curve_peak_matches_the_previous_scalar_estimator(
    sample_rate_hz: int, delay_estimation: DelayEstimationConfiguration
) -> None:
    delay_samples = 250
    signal_1, signal_2 = make_delayed_pair(sample_rate_hz, delay_samples, seed=3)
    scalar_delay_s = compute_generalised_cross_correlation_phat(
        signal_1,
        signal_2,
        sample_rate_hz,
        CURVE_MAXIMUM_LAG_S,
        delay_estimation.interpolation_factor,
    )
    curve = compute_generalized_cross_correlation(
        signal_1,
        signal_2,
        (0, 1),
        sample_rate_hz,
        CURVE_MAXIMUM_LAG_S,
        build_correlation_configuration(),
    )
    curve_delay_s = estimate_delay_from_correlation_curve(
        curve, PARABOLIC_INTERPOLATION, 2.0, delay_estimation.interpolation_factor
    ).delay_s
    assert curve_delay_s == pytest.approx(scalar_delay_s, abs=2.0 / sample_rate_hz)


def test_the_whitening_ignores_noise_outside_the_analysis_band(
    sample_rate_hz: int,
) -> None:
    delay_samples = 300
    signal_1, signal_2 = make_delayed_pair(sample_rate_hz, delay_samples, seed=5)
    generator = np.random.default_rng(9)
    times_s = np.arange(signal_1.size) / sample_rate_hz
    out_of_band = 20.0 * np.sin(2.0 * np.pi * 20000.0 * times_s)
    noisy_1 = signal_1 + out_of_band + generator.standard_normal(signal_1.size)
    noisy_2 = signal_2 + out_of_band[: signal_2.size]

    configuration = build_correlation_configuration()
    clean_delay_s = estimate_delay_from_correlation_curve(
        compute_generalized_cross_correlation(
            signal_1,
            signal_2,
            (0, 1),
            sample_rate_hz,
            CURVE_MAXIMUM_LAG_S,
            configuration,
        ),
        PARABOLIC_INTERPOLATION,
        2.0,
        16,
    ).delay_s
    noisy_delay_s = estimate_delay_from_correlation_curve(
        compute_generalized_cross_correlation(
            noisy_1,
            noisy_2,
            (0, 1),
            sample_rate_hz,
            CURVE_MAXIMUM_LAG_S,
            configuration,
        ),
        PARABOLIC_INTERPOLATION,
        2.0,
        16,
    ).delay_s
    assert noisy_delay_s == pytest.approx(clean_delay_s, abs=2.0 / sample_rate_hz)


def test_the_envelope_removes_the_sign_changes_a_band_pass_introduces(
    sample_rate_hz: int,
) -> None:
    signal_1, signal_2 = make_delayed_pair(sample_rate_hz, 300, seed=7)
    plain = compute_generalized_cross_correlation(
        signal_1,
        signal_2,
        (0, 1),
        sample_rate_hz,
        CURVE_MAXIMUM_LAG_S,
        build_correlation_configuration(use_analytic_envelope=False),
    )
    enveloped = compute_generalized_cross_correlation(
        signal_1,
        signal_2,
        (0, 1),
        sample_rate_hz,
        CURVE_MAXIMUM_LAG_S,
        build_correlation_configuration(use_analytic_envelope=True),
    )
    assert np.any(plain.values < 0.0)
    assert np.all(enveloped.values >= 0.0)
    assert enveloped.lags_s[int(np.argmax(enveloped.values))] == pytest.approx(
        plain.lags_s[int(np.argmax(np.abs(plain.values)))],
        abs=2.0 / sample_rate_hz,
    )


def test_windowed_accumulation_sharpens_the_peak(sample_rate_hz: int) -> None:
    signal_1, signal_2 = make_delayed_pair(sample_rate_hz, 300, seed=11)
    curve = accumulate_generalized_cross_correlation_over_windows(
        signal_1,
        signal_2,
        (0, 1),
        sample_rate_hz,
        CURVE_MAXIMUM_LAG_S,
        build_correlation_configuration(use_analytic_envelope=True),
    )
    assert curve.window_count > 10
    estimate = estimate_delay_from_correlation_curve(
        curve, EXPONENTIAL_INTERPOLATION, 2.0, 16
    )
    assert estimate.delay_s == pytest.approx(300 / sample_rate_hz, abs=1e-4)
    assert estimate.peak_to_median_ratio > 5.0


def test_a_record_shorter_than_one_window_is_rejected(sample_rate_hz: int) -> None:
    with pytest.raises(ValueError, match="shorter than one accumulation window"):
        accumulate_generalized_cross_correlation_over_windows(
            np.zeros(100),
            np.zeros(100),
            (0, 1),
            sample_rate_hz,
            CURVE_MAXIMUM_LAG_S,
            build_correlation_configuration(),
        )


def test_one_curve_is_returned_per_receiver_pair(sample_rate_hz: int) -> None:
    signal_1, signal_2 = make_delayed_pair(sample_rate_hz, 200, seed=13)
    channels = np.stack(
        (signal_1[: signal_2.size], signal_2, signal_2, signal_1[: signal_2.size])
    )
    receiver_pairs = enumerate_receiver_pairs(4)
    curves = compute_pair_correlation_curves(
        channels,
        receiver_pairs,
        CURVE_MAXIMUM_LAG_S,
        sample_rate_hz,
        build_correlation_configuration(),
    )
    assert len(curves) == receiver_pairs.shape[0]
    assert [curve.receiver_pair for curve in curves] == [
        (int(pair[0]), int(pair[1])) for pair in receiver_pairs
    ]


def test_exponential_interpolation_refines_a_gaussian_peak_exactly() -> None:
    offsets = np.arange(-5.0, 6.0)
    values = np.exp(-0.5 * ((offsets - 0.3) / 2.0) ** 2)
    refined = refine_peak_index(
        values, int(np.argmax(values)), EXPONENTIAL_INTERPOLATION
    )
    assert refined == pytest.approx(5.0 + 0.3, abs=1e-9)


def test_an_unknown_interpolation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown peak interpolation"):
        refine_peak_index(np.array([0.0, 1.0, 0.0]), 1, "nonsense")


def test_reading_near_a_prediction_finds_the_quieter_of_two_peaks(
    sample_rate_hz: int,
) -> None:
    lags_s = np.arange(-1000, 1001, dtype=np.float64) / sample_rate_hz
    loud_delay_s = 400.0 / sample_rate_hz
    quiet_delay_s = -300.0 / sample_rate_hz
    width_s = 4.0 / sample_rate_hz
    curve = GeneralizedCrossCorrelationCurve(
        receiver_pair=(0, 1),
        lags_s=lags_s,
        values=np.exp(-0.5 * ((lags_s - loud_delay_s) / width_s) ** 2)
        + 0.3 * np.exp(-0.5 * ((lags_s - quiet_delay_s) / width_s) ** 2),
        sample_rate_hz=sample_rate_hz,
        window_count=1,
        effective_bandwidth_hz=5000.0,
    )
    global_estimate = estimate_delay_from_correlation_curve(
        curve, PARABOLIC_INTERPOLATION, 1.0, 16
    )
    local_estimate = estimate_delay_near_prediction(
        curve,
        quiet_delay_s,
        50.0 / sample_rate_hz,
        PARABOLIC_INTERPOLATION,
        1.0,
        16,
    )
    assert global_estimate.delay_s == pytest.approx(loud_delay_s, abs=1e-5)
    assert local_estimate.delay_s == pytest.approx(quiet_delay_s, abs=1e-5)


def test_zero_coherence_gives_a_finite_variance(sample_rate_hz: int) -> None:
    """A fully deflated pair must weight itself out, not produce an infinity.

    Deflation can leave a curve whose peak is no taller than its own median,
    which reads as zero coherence. Dividing by that used to raise and return an
    infinite variance, which then propagated into the weighted position fit.
    """
    variance_s2 = compute_delay_variance_s2(5000.0, 0.0, sample_rate_hz, 1.0, 16)
    assert np.isfinite(variance_s2)
    assert variance_s2 > compute_delay_variance_s2(5000.0, 0.9, sample_rate_hz, 1.0, 16)
