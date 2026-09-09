import numpy as np
import pytest

from audio.band_level_meter import (
    compute_band_level_db,
    compute_window_start_indices,
    make_analysis_window,
)

WINDOW_SAMPLE_COUNT = 2048


def test_analysis_window_has_the_requested_length() -> None:
    assert make_analysis_window(WINDOW_SAMPLE_COUNT).size == WINDOW_SAMPLE_COUNT


def test_analysis_window_tapers_to_its_edges() -> None:
    window = make_analysis_window(WINDOW_SAMPLE_COUNT)
    assert window[0] == pytest.approx(0.0, abs=1e-12)
    assert window[-1] == pytest.approx(0.0, abs=1e-12)
    assert window[WINDOW_SAMPLE_COUNT // 2] == pytest.approx(1.0, abs=1e-3)


def test_level_recovers_a_scaling_in_decibels() -> None:
    window = make_analysis_window(WINDOW_SAMPLE_COUNT)
    signal = np.random.default_rng(1).normal(size=WINDOW_SAMPLE_COUNT)
    level_db = compute_band_level_db(signal, window)
    scaled_level_db = compute_band_level_db(2.0 * signal, window)
    assert scaled_level_db - level_db == pytest.approx(20.0 * np.log10(2.0), abs=1e-6)


def test_level_divides_out_the_window_power() -> None:
    window = make_analysis_window(WINDOW_SAMPLE_COUNT)
    signal = np.ones(WINDOW_SAMPLE_COUNT)
    assert compute_band_level_db(signal, window) == pytest.approx(0.0, abs=1e-6)


def test_silence_is_floored_rather_than_infinite() -> None:
    window = make_analysis_window(WINDOW_SAMPLE_COUNT)
    assert np.isfinite(compute_band_level_db(np.zeros(WINDOW_SAMPLE_COUNT), window))


def test_window_starts_step_by_the_hop() -> None:
    assert compute_window_start_indices(0, 100, 40, 20) == [0, 20, 40, 60]


def test_window_starts_never_run_past_the_last_index() -> None:
    starts = compute_window_start_indices(0, 100, 40, 20)
    assert all(start + 40 <= 100 for start in starts)


def test_window_starts_respect_the_first_index() -> None:
    assert compute_window_start_indices(10, 100, 40, 20) == [10, 30, 50]


def test_range_too_short_for_one_window_yields_nothing() -> None:
    assert compute_window_start_indices(0, 30, 40, 20) == []


def test_non_positive_hop_is_rejected() -> None:
    with pytest.raises(ValueError, match="hop_sample_count must be positive"):
        compute_window_start_indices(0, 100, 40, 0)
