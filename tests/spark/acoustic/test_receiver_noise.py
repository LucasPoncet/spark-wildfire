import numpy as np
import pytest

from spark.acoustic.receiver_noise import (
    add_white_noise_at_snr_db,
    add_white_noise_to_receiver_signals,
)


def make_signal(sample_count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=sample_count)


def test_noise_reaches_the_requested_signal_to_noise_ratio(sample_rate_hz: int) -> None:
    signal = make_signal(sample_rate_hz)
    noisy = add_white_noise_at_snr_db(signal, 20.0, np.random.default_rng(3))
    measured_snr_db = 10.0 * np.log10(
        np.mean(signal**2) / np.mean((noisy - signal) ** 2)
    )
    assert measured_snr_db == pytest.approx(20.0, abs=0.5)


def test_a_lower_ratio_adds_more_noise(sample_rate_hz: int) -> None:
    signal = make_signal(sample_rate_hz)
    quiet = add_white_noise_at_snr_db(signal, 30.0, np.random.default_rng(0))
    loud = add_white_noise_at_snr_db(signal, 0.0, np.random.default_rng(0))
    assert np.mean((loud - signal) ** 2) > np.mean((quiet - signal) ** 2)


def test_noise_preserves_the_signal_length(sample_rate_hz: int) -> None:
    signal = make_signal(sample_rate_hz)
    assert (
        add_white_noise_at_snr_db(signal, 10.0, np.random.default_rng(0)).size
        == signal.size
    )


def test_a_silent_signal_is_returned_unchanged() -> None:
    silence = np.zeros(1000)
    assert np.allclose(
        add_white_noise_at_snr_db(silence, 10.0, np.random.default_rng(0)), silence
    )


def test_the_same_seed_reproduces_the_same_noise(sample_rate_hz: int) -> None:
    signal = make_signal(sample_rate_hz)
    first = add_white_noise_at_snr_db(signal, 10.0, np.random.default_rng(7))
    second = add_white_noise_at_snr_db(signal, 10.0, np.random.default_rng(7))
    assert np.allclose(first, second)


def test_every_channel_receives_noise(sample_rate_hz: int) -> None:
    channels = np.stack([make_signal(sample_rate_hz, seed) for seed in range(3)])
    noisy = add_white_noise_to_receiver_signals(
        channels, 10.0, np.random.default_rng(0)
    )
    assert noisy.shape == channels.shape
    assert np.all(np.any(noisy != channels, axis=1))


def test_channel_noise_is_drawn_independently(sample_rate_hz: int) -> None:
    channels = np.stack(
        [make_signal(sample_rate_hz, 0), make_signal(sample_rate_hz, 0)]
    )
    noisy = add_white_noise_to_receiver_signals(
        channels, 10.0, np.random.default_rng(0)
    )
    assert not np.allclose(noisy[0] - channels[0], noisy[1] - channels[1])
