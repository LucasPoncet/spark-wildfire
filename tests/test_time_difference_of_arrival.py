import numpy as np
import pytest

from src.audio.signal_alignment import (
    align_channel_pair,
    compute_valid_index_range,
    shift_signal_by_samples,
)
from src.config.simulation_configuration import DelayEstimationConfiguration
from src.spark.inverse.time_difference_of_arrival import (
    estimate_time_difference_of_arrival,
)

SEARCH_WINDOW_S: float = 0.02


def test_shift_delays_a_signal_for_a_positive_shift() -> None:
    signal = np.arange(5.0)
    assert np.allclose(shift_signal_by_samples(signal, 2), [0.0, 0.0, 0.0, 1.0, 2.0])


def test_shift_advances_a_signal_for_a_negative_shift() -> None:
    signal = np.arange(5.0)
    assert np.allclose(shift_signal_by_samples(signal, -2), [2.0, 3.0, 4.0, 0.0, 0.0])


def test_valid_index_range_excludes_the_zero_filled_region() -> None:
    assert compute_valid_index_range(100, 100, 10) == (10, 100)
    assert compute_valid_index_range(100, 100, -10) == (0, 90)


def test_alignment_recovers_a_known_delay() -> None:
    base = np.random.default_rng(0).normal(size=1000)
    delayed = shift_signal_by_samples(base, 40)
    aligned_1, aligned_2 = align_channel_pair(delayed, base, 40)
    assert np.allclose(aligned_1, aligned_2)


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
