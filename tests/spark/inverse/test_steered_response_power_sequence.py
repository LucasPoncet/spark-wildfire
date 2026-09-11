"""Framing the waveforms and folding the resulting maps back into images.

The framing arithmetic is silent when wrong: a hop computed one sample out
still produces a plausible-looking sequence, and every quantity downstream is a
regression against the frame times.
"""

import numpy as np
import pytest

from src.config.multi_source_localization_configuration import (
    CorrelationConfiguration,
    SteeredResponsePowerConfiguration,
)
from src.spark.inverse.receiver_pair_index import enumerate_receiver_pairs
from src.spark.inverse.steered_response_power import build_candidate_grid_xyz
from src.spark.inverse.steered_response_power_sequence import (
    SteeredResponsePowerSequence,
    compute_frame_start_indices,
    compute_imaging_grid_shape,
    compute_steered_response_power_sequence,
    reshape_sequence_map_to_grid,
)

SAMPLE_RATE_HZ: int = 8000
DOMAIN_EXTENT_M: float = 20.0
GRID_SPACING_M: float = 1.0
SPEED_OF_SOUND_M_PER_S: float = 340.0
SOURCE_XY_M = np.array([13.0, 8.0])
RECEIVERS_XYZ_M = np.array(
    [
        [2.0, 2.0, 1.0],
        [18.0, 2.0, 1.0],
        [18.0, 18.0, 1.0],
        [2.0, 18.0, 1.0],
        [10.0, 1.0, 1.0],
    ]
)

CORRELATION = CorrelationConfiguration(
    lowest_frequency_hz=200.0,
    highest_frequency_hz=3500.0,
    phase_transform_exponent=0.0,
    phase_transform_regularization=1e-8,
    use_analytic_envelope=True,
    window_duration_s=0.1,
    window_overlap=0.5,
    maximum_absolute_lag_margin=1.1,
    peak_interpolation="exponential",
)
STEERED_RESPONSE_POWER = SteeredResponsePowerConfiguration(
    pooling="mean",
    pairwise_combinator="sum",
    coarse_grid_spacing_m=GRID_SPACING_M,
    refinement_levels=1,
    refinement_factor=2,
    refinement_retained_fraction=0.1,
    low_frequency_seed_hz=2000.0,
    candidate_height_m=0.5,
)


def render_delayed_channels(duration_s: float) -> np.ndarray:
    sample_count = round(duration_s * SAMPLE_RATE_HZ)
    source = np.random.default_rng(0).standard_normal(sample_count + 2048)
    distances_m = np.linalg.norm(
        RECEIVERS_XYZ_M - np.array([SOURCE_XY_M[0], SOURCE_XY_M[1], 0.5]), axis=1
    )
    delays_samples = np.round(
        distances_m / SPEED_OF_SOUND_M_PER_S * SAMPLE_RATE_HZ
    ).astype(int)
    offset = int(np.max(delays_samples))
    return np.stack(
        [
            source[offset - delay : offset - delay + sample_count] / distance_m
            for delay, distance_m in zip(delays_samples, distances_m, strict=True)
        ]
    )


def test_frame_starts_advance_by_the_hop() -> None:
    starts = compute_frame_start_indices(10 * SAMPLE_RATE_HZ, 4.0, 2.0, SAMPLE_RATE_HZ)
    assert np.all(np.diff(starts) == 2 * SAMPLE_RATE_HZ)


def test_a_trailing_partial_frame_is_dropped() -> None:
    starts = compute_frame_start_indices(10 * SAMPLE_RATE_HZ, 4.0, 2.0, SAMPLE_RATE_HZ)
    assert int(starts[-1]) + 4 * SAMPLE_RATE_HZ <= 10 * SAMPLE_RATE_HZ
    assert (
        int(starts[-1]) + 4 * SAMPLE_RATE_HZ > 10 * SAMPLE_RATE_HZ - 2 * SAMPLE_RATE_HZ
    )


def test_a_signal_shorter_than_one_frame_yields_no_frame() -> None:
    assert (
        compute_frame_start_indices(SAMPLE_RATE_HZ, 4.0, 2.0, SAMPLE_RATE_HZ).size == 0
    )


def test_a_signal_exactly_one_frame_long_yields_one_frame() -> None:
    starts = compute_frame_start_indices(4 * SAMPLE_RATE_HZ, 4.0, 2.0, SAMPLE_RATE_HZ)
    assert starts.size == 1
    assert int(starts[0]) == 0


def test_a_non_positive_frame_duration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must both be positive"):
        compute_frame_start_indices(SAMPLE_RATE_HZ, 0.0, 2.0, SAMPLE_RATE_HZ)


def test_the_grid_shape_matches_the_grid_that_is_actually_built() -> None:
    grid = build_candidate_grid_xyz(
        DOMAIN_EXTENT_M, 0.5 * DOMAIN_EXTENT_M, GRID_SPACING_M, 0.5
    )
    row_count, column_count = compute_imaging_grid_shape(
        DOMAIN_EXTENT_M, 0.5 * DOMAIN_EXTENT_M, GRID_SPACING_M
    )
    assert row_count * column_count == grid.positions_xyz_m.shape[0]


def test_a_non_positive_grid_spacing_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_imaging_grid_shape(DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, 0.0)


def build_sequence() -> SteeredResponsePowerSequence:
    return compute_steered_response_power_sequence(
        render_delayed_channels(1.2),
        RECEIVERS_XYZ_M,
        enumerate_receiver_pairs(RECEIVERS_XYZ_M.shape[0]),
        0.4,
        0.4,
        SAMPLE_RATE_HZ,
        SPEED_OF_SOUND_M_PER_S,
        DOMAIN_EXTENT_M,
        DOMAIN_EXTENT_M,
        GRID_SPACING_M,
        CORRELATION,
        STEERED_RESPONSE_POWER,
    )


def test_the_sequence_has_one_map_per_frame_over_a_shared_grid() -> None:
    sequence = build_sequence()
    assert sequence.frame_count == 3
    assert sequence.maps.shape == (
        3,
        sequence.candidate_positions_xyz_m.shape[0],
    )
    assert sequence.grid_shape[0] * sequence.grid_shape[1] == sequence.maps.shape[1]


def test_every_frame_is_timed_at_its_own_centre() -> None:
    sequence = build_sequence()
    assert sequence.times_s == pytest.approx([0.2, 0.6, 1.0])


def test_every_frame_of_a_stationary_source_peaks_on_that_source() -> None:
    sequence = build_sequence()
    for frame_index in range(sequence.frame_count):
        peak_xy_m = sequence.candidate_positions_xyz_m[
            int(np.argmax(sequence.maps[frame_index])), :2
        ]
        assert float(np.linalg.norm(peak_xy_m - SOURCE_XY_M)) <= 2.0 * GRID_SPACING_M


def test_a_frame_folds_back_into_the_grid_it_came_from() -> None:
    sequence = build_sequence()
    image = reshape_sequence_map_to_grid(sequence, 0)
    assert image.shape == sequence.grid_shape
    assert image.ravel() == pytest.approx(sequence.maps[0])


def test_folding_a_frame_that_does_not_exist_is_rejected() -> None:
    sequence = build_sequence()
    with pytest.raises(IndexError, match="outside a sequence"):
        reshape_sequence_map_to_grid(sequence, sequence.frame_count)


def test_signals_shorter_than_one_frame_are_rejected() -> None:
    with pytest.raises(ValueError, match="shorter than one frame"):
        compute_steered_response_power_sequence(
            render_delayed_channels(0.2),
            RECEIVERS_XYZ_M,
            enumerate_receiver_pairs(RECEIVERS_XYZ_M.shape[0]),
            0.4,
            0.4,
            SAMPLE_RATE_HZ,
            SPEED_OF_SOUND_M_PER_S,
            DOMAIN_EXTENT_M,
            DOMAIN_EXTENT_M,
            GRID_SPACING_M,
            CORRELATION,
            STEERED_RESPONSE_POWER,
        )
