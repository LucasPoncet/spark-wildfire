import numpy as np
import pytest

from src.audio.codec_bandwidth_detector import (
    compute_long_term_average_spectrum,
    detect_codec_cutoff_hz,
)

WINDOW_SAMPLE_COUNT: int = 8192
OVERLAP_FRACTION: float = 0.5
FLOOR_DROP_DB: float = 50.0
LEAKAGE_TOLERANCE_BINS: int = 4


def make_brickwalled_noise(
    sample_rate_hz: int, duration_s: float, cutoff_hz: float, seed: int = 0
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    sample_count = round(duration_s * sample_rate_hz)
    spectrum = np.fft.rfft(generator.standard_normal(sample_count))
    frequencies_hz = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate_hz)
    spectrum[frequencies_hz > cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, sample_count)


def test_long_term_average_spectrum_has_one_level_per_bin(
    sample_rate_hz: int,
) -> None:
    samples = make_brickwalled_noise(sample_rate_hz, 2.0, 8000.0)
    frequencies_hz, level_db = compute_long_term_average_spectrum(
        samples, sample_rate_hz, WINDOW_SAMPLE_COUNT, OVERLAP_FRACTION
    )
    assert frequencies_hz.shape == level_db.shape
    assert frequencies_hz.size == WINDOW_SAMPLE_COUNT // 2 + 1
    assert frequencies_hz[-1] == pytest.approx(0.5 * sample_rate_hz)


def test_brickwall_cutoff_is_recovered_within_the_window_skirt(
    sample_rate_hz: int,
) -> None:
    """A brickwall is recovered to the resolution the analysis window allows.

    Not to one bin: the Hann window spreads a step edge over its own skirt, so
    the measured spectrum falls from the in-band level to fifty decibels below
    it over about four bins however sharp the wall really is. Asserting one bin
    would be asserting something no windowed spectrum can deliver.
    """
    true_cutoff_hz = 11025.0
    samples = make_brickwalled_noise(sample_rate_hz, 4.0, true_cutoff_hz)
    detected_hz = detect_codec_cutoff_hz(
        samples,
        sample_rate_hz,
        WINDOW_SAMPLE_COUNT,
        OVERLAP_FRACTION,
        FLOOR_DROP_DB,
    )
    bin_width_hz = sample_rate_hz / WINDOW_SAMPLE_COUNT
    assert abs(detected_hz - true_cutoff_hz) <= LEAKAGE_TOLERANCE_BINS * bin_width_hz


def test_a_lower_brickwall_is_detected_lower(sample_rate_hz: int) -> None:
    low_cutoff_hz = detect_codec_cutoff_hz(
        make_brickwalled_noise(sample_rate_hz, 4.0, 6000.0),
        sample_rate_hz,
        WINDOW_SAMPLE_COUNT,
        OVERLAP_FRACTION,
        FLOOR_DROP_DB,
    )
    high_cutoff_hz = detect_codec_cutoff_hz(
        make_brickwalled_noise(sample_rate_hz, 4.0, 15500.0),
        sample_rate_hz,
        WINDOW_SAMPLE_COUNT,
        OVERLAP_FRACTION,
        FLOOR_DROP_DB,
    )
    assert low_cutoff_hz < high_cutoff_hz
    assert abs(low_cutoff_hz - 6000.0) <= LEAKAGE_TOLERANCE_BINS * (
        sample_rate_hz / WINDOW_SAMPLE_COUNT
    )


def test_full_band_noise_reports_a_cutoff_near_nyquist(sample_rate_hz: int) -> None:
    generator = np.random.default_rng(1)
    samples = generator.standard_normal(4 * sample_rate_hz)
    detected_hz = detect_codec_cutoff_hz(
        samples,
        sample_rate_hz,
        WINDOW_SAMPLE_COUNT,
        OVERLAP_FRACTION,
        FLOOR_DROP_DB,
    )
    assert detected_hz > 0.45 * sample_rate_hz


def test_a_recording_shorter_than_one_window_is_rejected(sample_rate_hz: int) -> None:
    with pytest.raises(ValueError, match="shorter than one analysis window"):
        compute_long_term_average_spectrum(
            np.zeros(100), sample_rate_hz, WINDOW_SAMPLE_COUNT, OVERLAP_FRACTION
        )


def test_an_out_of_range_overlap_is_rejected(sample_rate_hz: int) -> None:
    with pytest.raises(ValueError, match=r"overlap_fraction must lie in \[0, 1\)"):
        compute_long_term_average_spectrum(
            np.zeros(2 * WINDOW_SAMPLE_COUNT), sample_rate_hz, WINDOW_SAMPLE_COUNT, 1.0
        )
