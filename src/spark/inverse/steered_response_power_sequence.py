"""A steered response power map per time frame, on one fixed grid.

The detection loop searches coarse-to-fine and keeps only the level it needed;
imaging wants the opposite — one grid, fine enough to carry the source
structure, evaluated identically at every frame so that frames can be
differenced and regressed against time.

Frames are cut from the receiver waveforms, not from the maps: a correlation
curve is accumulated over the windows inside one frame, so the frame length
sets both how much the front smears and how many windows average into each
curve. Those pull against each other and the balance is a measurement, not a
default — the frame duration belongs in configuration for that reason.
"""

from dataclasses import dataclass

import numpy as np

from src.config.multi_source_localization_configuration import (
    CorrelationConfiguration,
    SteeredResponsePowerConfiguration,
)
from src.spark.inverse.receiver_pair_index import compute_maximum_absolute_lag_s
from src.spark.inverse.steered_response_power import (
    CandidateGrid,
    build_candidate_grid_xyz,
    compute_map_over_grid,
)
from src.spark.inverse.time_difference_of_arrival import compute_pair_correlation_curves
from src.utils.array_types import Float64Array, Int64Array


@dataclass(frozen=True)
class SteeredResponsePowerSequence:
    """One map per frame over a shared grid, with the clock they sit on.

    Attributes:
        maps: Maps of shape `(n_frames, n_cells)`, in frame order.
        times_s: Centre time of each frame, shape `(n_frames,)`. The centre
            rather than the start, because a frame describes the interval it
            covers and the centre is the instant a linear fit over frames is
            unbiased about.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        grid_shape: `(n_y, n_x)` of that grid.
        grid_spacing_m: Cell side length, in metres.
    """

    maps: Float64Array
    times_s: Float64Array
    candidate_positions_xyz_m: Float64Array
    grid_shape: tuple[int, int]
    grid_spacing_m: float

    @property
    def frame_count(self) -> int:
        """How many frames the sequence holds."""
        return int(np.asarray(self.maps).shape[0])


def compute_imaging_grid_shape(
    domain_extent_x_m: float, domain_extent_y_m: float, spacing_m: float
) -> tuple[int, int]:
    """Rows and columns of the grid `build_candidate_grid_xyz` lays down.

    Mirrors that function's own tiling rather than re-deriving it, so a map can
    always be folded back into an image.

    Args:
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        spacing_m: Cell side length, in metres.

    Returns:
        `(n_y, n_x)`.

    Raises:
        ValueError: If the spacing is not positive.
    """
    if spacing_m <= 0.0:
        raise ValueError("candidate grid spacing must be positive")
    column_count = int(np.arange(0.5 * spacing_m, domain_extent_x_m, spacing_m).size)
    row_count = int(np.arange(0.5 * spacing_m, domain_extent_y_m, spacing_m).size)
    return row_count, column_count


def compute_frame_start_indices(
    sample_count: int, frame_duration_s: float, frame_hop_s: float, sample_rate_hz: int
) -> Int64Array:
    """Sample index each frame starts at.

    A trailing partial frame is dropped rather than zero-padded, matching how
    `recording_segmenter` treats a partial segment: a short frame averages
    fewer correlation windows and would sit on the same axis as full ones
    without anything marking it as noisier.

    Args:
        sample_count: Length of the receiver channels, in samples.
        frame_duration_s: Length of one frame, in seconds.
        frame_hop_s: Advance between consecutive frame starts, in seconds.
        sample_rate_hz: Sample rate in hertz.

    Returns:
        Start indices, shape `(n_frames,)`, possibly empty when the signal is
        shorter than one frame.

    Raises:
        ValueError: If the duration or hop is not positive.
    """
    if frame_duration_s <= 0.0 or frame_hop_s <= 0.0:
        raise ValueError("frame duration and hop must both be positive")
    frame_sample_count = round(frame_duration_s * sample_rate_hz)
    hop_sample_count = max(1, round(frame_hop_s * sample_rate_hz))
    if frame_sample_count > sample_count:
        return np.empty(0, dtype=np.int64)
    last_start = sample_count - frame_sample_count
    return np.arange(0, last_start + 1, hop_sample_count, dtype=np.int64)


def compute_steered_response_power_sequence(
    receiver_signals: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    frame_duration_s: float,
    frame_hop_s: float,
    sample_rate_hz: int,
    speed_of_sound_m_per_s: float,
    domain_extent_x_m: float,
    domain_extent_y_m: float,
    grid_spacing_m: float,
    correlation_configuration: CorrelationConfiguration,
    steered_response_power_configuration: SteeredResponsePowerConfiguration,
) -> SteeredResponsePowerSequence:
    """Builds one map per frame of the receiver waveforms.

    The domain extents and grid spacing are arguments rather than being read
    off the search configuration, because the imaging grid is deliberately
    finer than the coarse search grid and the two must be free to differ.

    Args:
        receiver_signals: Channels, shape `(n_receivers, n_samples)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        frame_duration_s: Length of one frame, in seconds.
        frame_hop_s: Advance between consecutive frames, in seconds.
        sample_rate_hz: Sample rate in hertz.
        speed_of_sound_m_per_s: Speed of sound the estimator assumes.
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        grid_spacing_m: Imaging cell side length, in metres.
        correlation_configuration: Whitening band, exponent and windowing.
        steered_response_power_configuration: Pooling, combinator and the
            candidate height.

    Returns:
        The sequence.

    Raises:
        ValueError: If the signals are shorter than one frame.
    """
    channels = np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    grid = build_candidate_grid_xyz(
        domain_extent_x_m,
        domain_extent_y_m,
        grid_spacing_m,
        steered_response_power_configuration.candidate_height_m,
    )
    frame_starts = compute_frame_start_indices(
        channels.shape[1], frame_duration_s, frame_hop_s, sample_rate_hz
    )
    if frame_starts.size == 0:
        raise ValueError(
            "the receiver signals are shorter than one frame, so no map can be built"
        )

    maximum_absolute_lag_s = compute_maximum_absolute_lag_s(
        receivers,
        receiver_pairs,
        speed_of_sound_m_per_s,
        correlation_configuration.maximum_absolute_lag_margin,
    )
    frame_sample_count = round(frame_duration_s * sample_rate_hz)
    maps = np.empty(
        (frame_starts.size, grid.positions_xyz_m.shape[0]), dtype=np.float64
    )
    for frame_index, start_index in enumerate(frame_starts):
        frame_channels = channels[
            :, int(start_index) : int(start_index) + frame_sample_count
        ]
        maps[frame_index] = compute_map_over_grid(
            compute_pair_correlation_curves(
                frame_channels,
                receiver_pairs,
                maximum_absolute_lag_s,
                sample_rate_hz,
                correlation_configuration,
            ),
            grid,
            receivers,
            receiver_pairs,
            speed_of_sound_m_per_s,
            steered_response_power_configuration,
        )

    times_s = frame_starts.astype(np.float64) / sample_rate_hz + 0.5 * frame_duration_s
    return SteeredResponsePowerSequence(
        maps=maps,
        times_s=times_s,
        candidate_positions_xyz_m=grid.positions_xyz_m,
        grid_shape=compute_imaging_grid_shape(
            domain_extent_x_m, domain_extent_y_m, grid_spacing_m
        ),
        grid_spacing_m=grid_spacing_m,
    )


def build_imaging_candidate_grid(
    domain_extent_x_m: float,
    domain_extent_y_m: float,
    grid_spacing_m: float,
    candidate_height_m: float,
) -> CandidateGrid:
    """The grid an imaging map and its point spread function share.

    Both sides of the covariance subtraction must be evaluated on the same
    cells, so the grid is built once here and handed to each.

    Args:
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        grid_spacing_m: Imaging cell side length, in metres.
        candidate_height_m: Height of the candidate plane, in metres.

    Returns:
        The grid.
    """
    return build_candidate_grid_xyz(
        domain_extent_x_m, domain_extent_y_m, grid_spacing_m, candidate_height_m
    )


def reshape_sequence_map_to_grid(
    sequence: SteeredResponsePowerSequence, time_index: int
) -> Float64Array:
    """Folds one frame of a sequence back into an image.

    Named for the sequence rather than sharing the plotter's
    `reshape_map_to_grid`, which infers the grid from the positions of a single
    map; here the shape is already known and carried on the sequence.

    Args:
        sequence: The sequence to read.
        time_index: Which frame to fold.

    Returns:
        The frame as an image of shape `(n_y, n_x)`.

    Raises:
        IndexError: If the frame index is out of range.
    """
    maps = np.asarray(sequence.maps, dtype=np.float64)
    if not -maps.shape[0] <= time_index < maps.shape[0]:
        raise IndexError(
            f"frame {time_index} is outside a sequence of {maps.shape[0]} frames"
        )
    return np.asarray(maps[time_index].reshape(sequence.grid_shape), dtype=np.float64)
