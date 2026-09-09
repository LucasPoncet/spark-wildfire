import numpy as np
import pytest

from src.audio.octave_band_filter import (
    apply_octave_bandpass,
    compute_octave_band_edges_hz,
)
from src.config.simulation_configuration import BandConfiguration
from src.spark.acoustic.free_field_propagation import (
    apply_free_field_propagation,
    render_receiver_signals,
)
from src.spark.acoustic.receiver_noise import add_white_noise_at_snr_db
from src.spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.utils.array_types import Float64Array

NEAR_RANGE_M: float = 30.0
FAR_RANGE_M: float = 70.0


def make_source_signal(
    sample_rate_hz: int, duration_s: float = 2.0, seed: int = 0
) -> Float64Array:
    return np.random.default_rng(seed).normal(size=int(duration_s * sample_rate_hz))


def compute_delay_samples(
    distance_m: float, atmosphere: AtmosphericConditions, sample_rate_hz: int
) -> int:
    return round(
        distance_m
        / compute_speed_of_sound_m_per_s(atmosphere.air_temperature_celsius)
        * sample_rate_hz
    )


def compute_arrival_slice(
    distance_m: float,
    sample_count: int,
    margin_samples: int,
    atmosphere: AtmosphericConditions,
    sample_rate_hz: int,
) -> slice:
    delay_samples = compute_delay_samples(distance_m, atmosphere, sample_rate_hz)
    return slice(
        delay_samples + margin_samples, delay_samples + sample_count - margin_samples
    )


def measure_band_level_difference_db(
    center_frequency_hz: float,
    atmosphere: AtmosphericConditions,
    bands: BandConfiguration,
    sample_rate_hz: int,
    reference_distance_m: float,
) -> float:
    signal = make_source_signal(sample_rate_hz)
    margin_samples = sample_rate_hz // 10
    levels = []
    for distance_m in (NEAR_RANGE_M, FAR_RANGE_M):
        band = apply_octave_bandpass(
            apply_free_field_propagation(
                signal, sample_rate_hz, distance_m, atmosphere, reference_distance_m
            ),
            center_frequency_hz,
            sample_rate_hz,
            bands.filter_order,
            bands.maximum_edge_fraction_of_nyquist,
        )
        levels.append(
            np.std(
                band[
                    compute_arrival_slice(
                        distance_m,
                        signal.size,
                        margin_samples,
                        atmosphere,
                        sample_rate_hz,
                    )
                ]
            )
        )
    return float(20.0 * np.log10(levels[0] / levels[1]))


def test_propagation_applies_the_expected_delay(
    atmosphere: AtmosphericConditions, sample_rate_hz: int, reference_distance_m: float
) -> None:
    impulse = np.zeros(sample_rate_hz)
    impulse[100] = 1.0
    distance_m = 34.0
    received = apply_free_field_propagation(
        impulse, sample_rate_hz, distance_m, atmosphere, reference_distance_m
    )
    expected_delay_samples = compute_delay_samples(
        distance_m, atmosphere, sample_rate_hz
    )
    assert int(np.argmax(np.abs(received))) == pytest.approx(
        100 + expected_delay_samples, abs=1.0
    )


def test_propagation_output_is_long_enough_to_hold_the_delayed_signal(
    atmosphere: AtmosphericConditions, sample_rate_hz: int, reference_distance_m: float
) -> None:
    signal = make_source_signal(sample_rate_hz, duration_s=0.2)
    distance_m = 100.0
    received = apply_free_field_propagation(
        signal, sample_rate_hz, distance_m, atmosphere, reference_distance_m
    )
    assert received.size >= signal.size + compute_delay_samples(
        distance_m, atmosphere, sample_rate_hz
    )


@pytest.mark.parametrize("center_frequency_hz", [125.0, 1000.0, 4000.0])
def test_band_level_difference_matches_the_forward_model(
    center_frequency_hz: float,
    atmosphere: AtmosphericConditions,
    bands: BandConfiguration,
    sample_rate_hz: int,
    reference_distance_m: float,
) -> None:
    measured_level_difference_db = measure_band_level_difference_db(
        center_frequency_hz, atmosphere, bands, sample_rate_hz, reference_distance_m
    )
    absorption_db_per_m = float(
        compute_absorption_coefficients_db_per_m(
            np.array([center_frequency_hz]), atmosphere
        )[0]
    )
    expected_level_difference_db = 20.0 * np.log10(
        FAR_RANGE_M / NEAR_RANGE_M
    ) + absorption_db_per_m * (FAR_RANGE_M - NEAR_RANGE_M)
    assert measured_level_difference_db == pytest.approx(
        expected_level_difference_db, abs=0.25
    )


def test_band_edge_absorption_brackets_the_measured_level_difference(
    atmosphere: AtmosphericConditions,
    bands: BandConfiguration,
    sample_rate_hz: int,
    reference_distance_m: float,
) -> None:
    center_frequency_hz = 4000.0
    measured_level_difference_db = measure_band_level_difference_db(
        center_frequency_hz, atmosphere, bands, sample_rate_hz, reference_distance_m
    )
    lower_edge_hz, upper_edge_hz = compute_octave_band_edges_hz(
        center_frequency_hz, sample_rate_hz, bands.maximum_edge_fraction_of_nyquist
    )
    edge_absorption_db_per_m = compute_absorption_coefficients_db_per_m(
        np.array([lower_edge_hz, upper_edge_hz]), atmosphere
    )
    geometric_level_difference_db = 20.0 * np.log10(FAR_RANGE_M / NEAR_RANGE_M)
    path_difference_m = FAR_RANGE_M - NEAR_RANGE_M
    assert (
        geometric_level_difference_db + edge_absorption_db_per_m[0] * path_difference_m
        <= measured_level_difference_db
        <= geometric_level_difference_db
        + edge_absorption_db_per_m[1] * path_difference_m
    )


def test_render_returns_one_channel_per_receiver_on_a_common_clock(
    atmosphere: AtmosphericConditions, sample_rate_hz: int, reference_distance_m: float
) -> None:
    signal = make_source_signal(sample_rate_hz, duration_s=0.5)
    receivers = np.array([[30.0, 0.0], [70.0, 0.0], [50.0, 100.0]])
    rendered = render_receiver_signals(
        signal,
        sample_rate_hz,
        np.array([50.0, 50.0]),
        receivers,
        atmosphere,
        reference_distance_m,
    )
    assert rendered.shape[0] == receivers.shape[0]
    assert rendered.shape[1] >= signal.size


def test_added_noise_reaches_the_requested_signal_to_noise_ratio(
    sample_rate_hz: int,
) -> None:
    signal = make_source_signal(sample_rate_hz, duration_s=1.0)
    noisy = add_white_noise_at_snr_db(signal, 20.0, np.random.default_rng(3))
    measured_snr_db = 10.0 * np.log10(
        np.mean(signal**2) / np.mean((noisy - signal) ** 2)
    )
    assert measured_snr_db == pytest.approx(20.0, abs=0.5)
