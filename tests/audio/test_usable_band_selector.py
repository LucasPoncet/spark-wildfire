import numpy as np
import pytest

from src.audio.octave_band_filter import (
    THIRD_OCTAVE_FRACTION_DENOMINATOR,
    build_fractional_octave_centre_frequencies_hz,
)
from src.audio.usable_band_selector import (
    filter_bands_by_signal_to_noise_ratio,
    select_usable_band_centres_hz,
)

CODEC_CUTOFF_MARGIN: float = 0.9


def test_a_15500_hz_cutoff_admits_10_khz_and_rejects_12500_hz() -> None:
    candidates_hz = build_fractional_octave_centre_frequencies_hz(
        250.0, 16000.0, THIRD_OCTAVE_FRACTION_DENOMINATOR
    )
    usable_hz = select_usable_band_centres_hz(
        candidates_hz,
        15500.0,
        CODEC_CUTOFF_MARGIN,
        THIRD_OCTAVE_FRACTION_DENOMINATOR,
    )
    assert 10000.0 in usable_hz
    assert 12500.0 not in usable_hz
    assert 16000.0 not in usable_hz


def test_every_usable_band_stays_below_the_margin() -> None:
    candidates_hz = build_fractional_octave_centre_frequencies_hz(
        250.0, 20000.0, THIRD_OCTAVE_FRACTION_DENOMINATOR
    )
    codec_cutoff_hz = 15500.0
    usable_hz = select_usable_band_centres_hz(
        candidates_hz,
        codec_cutoff_hz,
        CODEC_CUTOFF_MARGIN,
        THIRD_OCTAVE_FRACTION_DENOMINATOR,
    )
    upper_edge_factor = 2.0 ** (1.0 / (2.0 * THIRD_OCTAVE_FRACTION_DENOMINATOR))
    assert usable_hz
    assert all(
        centre_hz * upper_edge_factor <= CODEC_CUTOFF_MARGIN * codec_cutoff_hz
        for centre_hz in usable_hz
    )


def test_the_selection_preserves_ascending_order() -> None:
    candidates_hz = build_fractional_octave_centre_frequencies_hz(
        250.0, 10000.0, THIRD_OCTAVE_FRACTION_DENOMINATOR
    )
    usable_hz = select_usable_band_centres_hz(
        candidates_hz, 15700.0, CODEC_CUTOFF_MARGIN, THIRD_OCTAVE_FRACTION_DENOMINATOR
    )
    assert list(usable_hz) == sorted(usable_hz)


def test_bands_are_kept_only_when_they_clear_the_noise_floor() -> None:
    levels_db = np.array([40.0, 20.0, 10.0], dtype=np.float64)
    floors_db = np.array([10.0, 10.0, 9.0], dtype=np.float64)
    usable = filter_bands_by_signal_to_noise_ratio(levels_db, floors_db, 6.0)
    assert list(usable) == [True, True, False]


def test_mismatched_level_and_floor_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="one noise floor level is required"):
        filter_bands_by_signal_to_noise_ratio(np.zeros(3), np.zeros(2), 6.0)
