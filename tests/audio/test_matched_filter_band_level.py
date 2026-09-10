import numpy as np
import pytest

from src.audio.matched_filter_band_level import (
    compute_band_bin_masks,
    compute_matched_filter_band_levels_db,
)
from src.audio.octave_band_filter import (
    THIRD_OCTAVE_FRACTION_DENOMINATOR,
    build_fractional_octave_centre_frequencies_hz,
    compute_fractional_octave_band_edges_hz,
)

WINDOW_SAMPLE_COUNT: int = 8192
OVERLAP_FRACTION: float = 0.5
MAXIMUM_EDGE_FRACTION_OF_NYQUIST: float = 0.99
RECOVERED_LEVEL_TOLERANCE_DB: float = 0.5


def build_bands(sample_rate_hz: int) -> tuple[np.ndarray, np.ndarray]:
    centres_hz = build_fractional_octave_centre_frequencies_hz(
        1000.0, 8000.0, THIRD_OCTAVE_FRACTION_DENOMINATOR
    )
    edges_hz = np.array(
        [
            compute_fractional_octave_band_edges_hz(
                centre_hz,
                THIRD_OCTAVE_FRACTION_DENOMINATOR,
                sample_rate_hz,
                MAXIMUM_EDGE_FRACTION_OF_NYQUIST,
            )
            for centre_hz in centres_hz
        ]
    )
    return np.asarray(centres_hz, dtype=np.float64), edges_hz


def test_band_masks_cover_the_bins_inside_each_band() -> None:
    frequencies_hz = np.arange(0.0, 1000.0, 10.0)
    edges_hz = np.array([[100.0, 200.0], [400.0, 800.0]])
    masks = compute_band_bin_masks(frequencies_hz, edges_hz)
    assert len(masks) == 2
    assert frequencies_hz[masks[0]].min() == 100.0
    assert frequencies_hz[masks[0]].max() == 200.0
    assert frequencies_hz[masks[1]].min() == 400.0


def test_a_malformed_edge_array_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(n_bands, 2\)"):
        compute_band_bin_masks(np.arange(10.0), np.array([1.0, 2.0, 3.0]))


def test_two_sources_at_known_gains_are_separated_by_the_matched_filter(
    sample_rate_hz: int,
) -> None:
    """A mixture is projected back onto one source, gain by gain.

    Each receiver carries the reference source at a known gain on top of an
    interferer that is uncorrelated with it, so the recovered level differences
    must come out at the ratios of those gains. What makes this work is that the
    interferer contributes nothing to the projection except finite-sample
    noise, which averages away over windows and bins.
    """
    generator = np.random.default_rng(0)
    sample_count = 8 * sample_rate_hz
    reference = generator.standard_normal(sample_count)
    interferer = generator.standard_normal(sample_count)

    receiver_gains = np.array([1.0, 0.5, 0.25])
    channels = np.stack(
        [gain * reference + 0.5 * interferer for gain in receiver_gains]
    )
    centres_hz, edges_hz = build_bands(sample_rate_hz)
    levels = compute_matched_filter_band_levels_db(
        reference,
        channels,
        centres_hz,
        edges_hz,
        sample_rate_hz,
        WINDOW_SAMPLE_COUNT,
        OVERLAP_FRACTION,
    )

    assert levels.levels_db.shape == (3, centres_hz.size)
    for receiver_index, gain in enumerate(receiver_gains[1:], start=1):
        expected_difference_db = 20.0 * np.log10(receiver_gains[0] / gain)
        recovered_difference_db = levels.levels_db[0] - levels.levels_db[receiver_index]
        assert np.allclose(
            recovered_difference_db,
            expected_difference_db,
            atol=RECOVERED_LEVEL_TOLERANCE_DB,
        )


def test_the_variance_is_reported_per_receiver_and_band(sample_rate_hz: int) -> None:
    generator = np.random.default_rng(1)
    reference = generator.standard_normal(2 * sample_rate_hz)
    channels = np.stack([reference, 0.5 * reference])
    centres_hz, edges_hz = build_bands(sample_rate_hz)
    levels = compute_matched_filter_band_levels_db(
        reference,
        channels,
        centres_hz,
        edges_hz,
        sample_rate_hz,
        WINDOW_SAMPLE_COUNT,
        OVERLAP_FRACTION,
    )
    assert levels.window_variance_db2.shape == levels.levels_db.shape
    assert levels.window_count > 1
    assert np.all(levels.window_variance_db2 >= 0.0)


def test_signals_shorter_than_one_window_are_rejected(sample_rate_hz: int) -> None:
    centres_hz, edges_hz = build_bands(sample_rate_hz)
    with pytest.raises(ValueError, match="shorter than one analysis window"):
        compute_matched_filter_band_levels_db(
            np.zeros(100),
            np.zeros((2, 100)),
            centres_hz,
            edges_hz,
            sample_rate_hz,
            WINDOW_SAMPLE_COUNT,
            OVERLAP_FRACTION,
        )


def test_mismatched_band_centres_and_edges_are_rejected(sample_rate_hz: int) -> None:
    centres_hz, edges_hz = build_bands(sample_rate_hz)
    with pytest.raises(ValueError, match="one pair of band edges"):
        compute_matched_filter_band_levels_db(
            np.zeros(2 * WINDOW_SAMPLE_COUNT),
            np.zeros((2, 2 * WINDOW_SAMPLE_COUNT)),
            centres_hz[:-1],
            edges_hz,
            sample_rate_hz,
            WINDOW_SAMPLE_COUNT,
            OVERLAP_FRACTION,
        )
