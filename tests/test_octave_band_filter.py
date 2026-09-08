import numpy as np
import pytest

from src.audio.band_level_meter import compute_band_level_db, make_analysis_window
from src.audio.octave_band_filter import apply_octave_bandpass, compute_octave_band_edges_hz
from src.config.simulation_configuration import BandConfiguration


def test_band_edges_span_one_octave(bands: BandConfiguration, sample_rate_hz: int) -> None:
    lower_hz, upper_hz = compute_octave_band_edges_hz(
        1000.0, sample_rate_hz, bands.maximum_edge_fraction_of_nyquist
    )
    assert upper_hz / lower_hz == pytest.approx(2.0)


def test_band_edges_are_clipped_below_nyquist(
    bands: BandConfiguration, sample_rate_hz: int
) -> None:
    lower_hz, upper_hz = compute_octave_band_edges_hz(
        16000.0, sample_rate_hz, bands.maximum_edge_fraction_of_nyquist
    )
    assert upper_hz < 0.5 * sample_rate_hz
    assert lower_hz < upper_hz


def test_band_edges_reject_a_band_above_nyquist(
    bands: BandConfiguration, sample_rate_hz: int
) -> None:
    with pytest.raises(ValueError):
        compute_octave_band_edges_hz(
            64000.0, sample_rate_hz, bands.maximum_edge_fraction_of_nyquist
        )


def test_every_configured_band_passes_a_tone_at_its_center_frequency(
    bands: BandConfiguration, sample_rate_hz: int
) -> None:
    time_s = np.arange(sample_rate_hz) / sample_rate_hz
    interior = slice(sample_rate_hz // 10, -sample_rate_hz // 10)
    for center_frequency_hz in bands.center_frequencies_hz:
        tone = np.sin(2.0 * np.pi * center_frequency_hz * time_s)
        filtered = apply_octave_bandpass(
            tone,
            center_frequency_hz,
            sample_rate_hz,
            bands.filter_order,
            bands.maximum_edge_fraction_of_nyquist,
        )
        gain_db = 20.0 * np.log10(np.std(filtered[interior]) / np.std(tone[interior]))
        assert gain_db > -1.0, f"band {center_frequency_hz} Hz attenuates its own centre"


def test_bandpass_rejects_a_tone_two_octaves_away(
    bands: BandConfiguration, sample_rate_hz: int
) -> None:
    center_frequency_hz = 1000.0
    time_s = np.arange(sample_rate_hz) / sample_rate_hz
    tone = np.sin(2.0 * np.pi * (center_frequency_hz / 4.0) * time_s)
    filtered = apply_octave_bandpass(
        tone,
        center_frequency_hz,
        sample_rate_hz,
        bands.filter_order,
        bands.maximum_edge_fraction_of_nyquist,
    )
    interior = slice(sample_rate_hz // 10, -sample_rate_hz // 10)
    gain_db = 20.0 * np.log10(np.std(filtered[interior]) / np.std(tone[interior]))
    assert gain_db < -30.0


def test_bandpass_is_zero_phase(bands: BandConfiguration, sample_rate_hz: int) -> None:
    signal = np.random.default_rng(0).normal(size=sample_rate_hz)
    filtered = apply_octave_bandpass(
        signal,
        1000.0,
        sample_rate_hz,
        bands.filter_order,
        bands.maximum_edge_fraction_of_nyquist,
    )
    correlation = np.correlate(filtered, filtered, mode="same")
    assert int(np.argmax(correlation)) == filtered.size // 2


def test_band_level_recovers_the_level_of_a_scaled_signal() -> None:
    window = make_analysis_window(2048)
    signal = np.random.default_rng(1).normal(size=2048)
    level_db = compute_band_level_db(signal, window)
    scaled_level_db = compute_band_level_db(2.0 * signal, window)
    assert scaled_level_db - level_db == pytest.approx(20.0 * np.log10(2.0), abs=1e-6)
