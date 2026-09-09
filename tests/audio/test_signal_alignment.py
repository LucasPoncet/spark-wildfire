import numpy as np
import pytest

from audio.signal_alignment import (
    align_channel_pair,
    compute_valid_index_range,
    shift_signal_by_samples,
)


def test_positive_shift_delays_the_signal() -> None:
    assert np.allclose(
        shift_signal_by_samples(np.arange(5.0), 2), [0.0, 0.0, 0.0, 1.0, 2.0]
    )


def test_negative_shift_advances_the_signal() -> None:
    assert np.allclose(
        shift_signal_by_samples(np.arange(5.0), -2), [2.0, 3.0, 4.0, 0.0, 0.0]
    )


def test_zero_shift_returns_an_equal_copy() -> None:
    signal = np.arange(5.0)
    shifted = shift_signal_by_samples(signal, 0)
    assert np.allclose(shifted, signal)
    assert shifted is not signal


def test_shift_beyond_the_length_empties_the_signal() -> None:
    assert np.allclose(shift_signal_by_samples(np.arange(5.0), 10), np.zeros(5))


def test_shift_preserves_length() -> None:
    signal = np.arange(5.0)
    assert shift_signal_by_samples(signal, 3).size == signal.size


def test_valid_range_excludes_the_zero_filled_head() -> None:
    assert compute_valid_index_range(100, 100, 10) == (10, 100)


def test_valid_range_excludes_the_zero_filled_tail() -> None:
    assert compute_valid_index_range(100, 100, -10) == (0, 90)


def test_valid_range_spans_everything_when_unshifted() -> None:
    assert compute_valid_index_range(100, 100, 0) == (0, 100)


def test_alignment_recovers_a_known_delay() -> None:
    base = np.random.default_rng(0).normal(size=1000)
    aligned_1, aligned_2 = align_channel_pair(
        shift_signal_by_samples(base, 40), base, 40
    )
    assert np.allclose(aligned_1, aligned_2)


def test_alignment_crops_both_channels_to_the_same_length() -> None:
    base = np.random.default_rng(0).normal(size=1000)
    aligned_1, aligned_2 = align_channel_pair(base, base, 40)
    assert aligned_1.size == aligned_2.size == 960


def test_alignment_without_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="no overlapping samples"):
        align_channel_pair(np.zeros(10), np.zeros(10), 20)
