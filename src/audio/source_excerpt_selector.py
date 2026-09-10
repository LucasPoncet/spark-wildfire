from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.audio.excerpt_coherence import (
    compute_maximum_off_diagonal_coherence,
    compute_pairwise_excerpt_coherence_matrix,
)
from src.utils.array_types import Float64Array
from src.utils.io.audio_file_reader import read_audio_file, read_audio_metadata

EXPLICIT_ASSIGNMENT: str = "explicit"
DISTINCT_PROVENANCE_ASSIGNMENT: str = "distinct_provenance"
MINIMUM_COHERENCE_ASSIGNMENT: str = "minimum_coherence"
DISTINCT_WINDOWS_ASSIGNMENT: str = "distinct_windows"


@dataclass(frozen=True)
class SourceExcerpt:
    """One recording excerpt standing in for one physical fire.

    Attributes:
        recording_path: File the excerpt was cut from.
        samples: Mono samples of the excerpt.
        sample_rate_hz: Sample rate in hertz.
        start_sample_index: Offset of the excerpt within the recording.
    """

    recording_path: Path
    samples: Float64Array
    sample_rate_hz: int
    start_sample_index: int


def read_leading_excerpt(
    recording_path: Path, clip_duration_s: float, as_mono: bool
) -> SourceExcerpt:
    """Reads the opening clip of one recording.

    Args:
        recording_path: File to read.
        clip_duration_s: Clip length in seconds.
        as_mono: Whether to average the channels down to one.

    Returns:
        The excerpt, cut to the requested duration.

    Raises:
        ValueError: If the recording is shorter than the requested clip.
    """
    samples, sample_rate_hz = read_audio_file(recording_path, as_mono)
    clip_sample_count = round(clip_duration_s * sample_rate_hz)
    if samples.size < clip_sample_count:
        raise ValueError(
            f"{recording_path} holds {samples.size / sample_rate_hz:.1f} s, "
            f"shorter than the requested {clip_duration_s} s excerpt"
        )
    return SourceExcerpt(
        recording_path=recording_path,
        samples=np.asarray(samples[:clip_sample_count], dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        start_sample_index=0,
    )


def read_excerpt_at_offset(
    recording_path: Path,
    start_sample_index: int,
    clip_duration_s: float,
    as_mono: bool,
) -> SourceExcerpt:
    """Reads one clip from anywhere inside a recording.

    Args:
        recording_path: File to read.
        start_sample_index: Offset of the clip within the recording.
        clip_duration_s: Clip length in seconds.
        as_mono: Whether to average the channels down to one.

    Returns:
        The excerpt, cut to the requested duration.

    Raises:
        ValueError: If the recording ends before the clip does.
    """
    samples, sample_rate_hz = read_audio_file(recording_path, as_mono)
    clip_sample_count = round(clip_duration_s * sample_rate_hz)
    stop_sample_index = start_sample_index + clip_sample_count
    if samples.size < stop_sample_index:
        raise ValueError(
            f"{recording_path} ends before a {clip_duration_s} s clip starting at "
            f"{start_sample_index / sample_rate_hz:.1f} s does"
        )
    return SourceExcerpt(
        recording_path=recording_path,
        samples=np.asarray(samples[start_sample_index:stop_sample_index]),
        sample_rate_hz=sample_rate_hz,
        start_sample_index=start_sample_index,
    )


def enumerate_pool_windows(
    recording_paths: list[Path], clip_duration_s: float, as_mono: bool
) -> list[SourceExcerpt]:
    """Cuts every recording in the pool into non-overlapping clips.

    A pool whose files are overlapping cuts of one looping recording offers only
    one usable excerpt per provenance at full clip length, which caps how many
    sources a scene can sound. Shorter, non-overlapping windows carry different
    content and so decorrelate where whole files do not, at the cost of fewer
    accumulation windows in the correlation.

    Args:
        recording_paths: Candidate recordings.
        clip_duration_s: Clip length in seconds.
        as_mono: Whether to average the channels down to one.

    Returns:
        Every whole clip of every recording, in file then offset order.
    """
    windows: list[SourceExcerpt] = []
    for path in recording_paths:
        samples, sample_rate_hz = read_audio_file(path, as_mono)
        clip_sample_count = round(clip_duration_s * sample_rate_hz)
        for start_sample_index in range(
            0, samples.size - clip_sample_count + 1, clip_sample_count
        ):
            windows.append(
                SourceExcerpt(
                    recording_path=path,
                    samples=np.asarray(
                        samples[
                            start_sample_index : start_sample_index + clip_sample_count
                        ]
                    ),
                    sample_rate_hz=sample_rate_hz,
                    start_sample_index=start_sample_index,
                )
            )
    return windows


def list_pool_recordings(
    recording_directory: Path, recording_filenames: tuple[str, ...]
) -> list[Path]:
    """Lists the recordings a run may draw its sources from.

    Args:
        recording_directory: Directory holding the pool.
        recording_filenames: Names to use, in order. Empty means every wav file
            in the directory, sorted by name.

    Returns:
        Paths to the candidate recordings.

    Raises:
        FileNotFoundError: If the directory or a named file does not exist.
    """
    if not recording_directory.is_dir():
        raise FileNotFoundError(f"recording directory not found: {recording_directory}")
    if not recording_filenames:
        return sorted(recording_directory.glob("*.wav"))
    paths = [recording_directory / filename for filename in recording_filenames]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"recording not found: {path}")
    return paths


def choose_least_coherent_indices(
    coherence_matrix: Float64Array, excerpt_count: int
) -> list[int]:
    """Greedily picks the subset whose worst mutual coherence is smallest.

    Starts from the least coherent pair and adds, each round, the candidate that
    raises the subset's worst coherence least.

    Args:
        coherence_matrix: Square pairwise coherence over the whole pool.
        excerpt_count: Size of the subset to pick.

    Returns:
        Indices into the pool, in the order they were chosen.

    Raises:
        ValueError: If the pool is smaller than the requested subset.
    """
    matrix = np.asarray(coherence_matrix, dtype=np.float64)
    pool_size = matrix.shape[0]
    if pool_size < excerpt_count:
        raise ValueError(
            f"the pool holds {pool_size} recordings, fewer than the "
            f"{excerpt_count} sources requested"
        )
    if excerpt_count == 1:
        return [0]

    masked = matrix + np.diag(np.full(pool_size, np.inf))
    first_index, second_index = np.unravel_index(np.argmin(masked), masked.shape)
    chosen = [int(first_index), int(second_index)]
    while len(chosen) < excerpt_count:
        worst_coherence = np.max(matrix[:, chosen], axis=1)
        worst_coherence[chosen] = np.inf
        chosen.append(int(np.argmin(worst_coherence)))
    return chosen


def select_source_excerpts(
    recording_paths: list[Path],
    source_count: int,
    clip_duration_s: float,
    assignment_policy: str,
    maximum_permitted_excerpt_coherence: float,
    selection_seed: int,
    as_mono: bool,
    recording_start_offsets_s: tuple[float, ...] = (),
) -> list[SourceExcerpt]:
    """Assigns one recording excerpt to each source and checks they are distinct.

    Args:
        recording_paths: Candidate recordings.
        source_count: Number of sources to feed.
        clip_duration_s: Excerpt length in seconds.
        assignment_policy: `explicit`, `distinct_provenance`, `minimum_coherence`
            or `distinct_windows`.
        maximum_permitted_excerpt_coherence: Largest mutual coherence a run accepts.
        selection_seed: Seed making a shuffled assignment reproducible.
        as_mono: Whether to average the channels down to one.
        recording_start_offsets_s: Offset into each named recording, used by the
            `explicit` policy to name a window set rather than search for one.
            Empty takes every excerpt from the start.

    Returns:
        One excerpt per source.

    Raises:
        ValueError: If the policy is unknown, the pool is too small, the sample
            rates disagree, or the excerpts are more mutually coherent than
            permitted.
    """
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if (
        assignment_policy != DISTINCT_WINDOWS_ASSIGNMENT
        and len(recording_paths) < source_count
    ):
        raise ValueError(
            f"the pool holds {len(recording_paths)} recordings, fewer than the "
            f"{source_count} sources requested"
        )

    if assignment_policy == EXPLICIT_ASSIGNMENT:
        selected_paths = recording_paths[:source_count]
        if not recording_start_offsets_s:
            excerpts = [
                read_leading_excerpt(path, clip_duration_s, as_mono)
                for path in selected_paths
            ]
        elif len(recording_start_offsets_s) < source_count:
            raise ValueError(
                f"the scene names {len(recording_start_offsets_s)} excerpt offsets, "
                f"fewer than the {source_count} sources requested"
            )
        else:
            excerpts = [
                read_excerpt_at_offset(
                    path,
                    round(offset_s * read_audio_metadata(path)[0]),
                    clip_duration_s,
                    as_mono,
                )
                for path, offset_s in zip(
                    selected_paths,
                    recording_start_offsets_s[:source_count],
                    strict=True,
                )
            ]
    elif assignment_policy == DISTINCT_PROVENANCE_ASSIGNMENT:
        order = np.random.default_rng(selection_seed).permutation(len(recording_paths))
        excerpts = [
            read_leading_excerpt(recording_paths[index], clip_duration_s, as_mono)
            for index in order[:source_count]
        ]
    elif assignment_policy in (
        MINIMUM_COHERENCE_ASSIGNMENT,
        DISTINCT_WINDOWS_ASSIGNMENT,
    ):
        candidates = (
            enumerate_pool_windows(recording_paths, clip_duration_s, as_mono)
            if assignment_policy == DISTINCT_WINDOWS_ASSIGNMENT
            else [
                read_leading_excerpt(path, clip_duration_s, as_mono)
                for path in recording_paths
            ]
        )
        indices = choose_least_coherent_indices(
            compute_pairwise_excerpt_coherence_matrix(
                [candidate.samples for candidate in candidates]
            ),
            source_count,
        )
        excerpts = [candidates[index] for index in indices]
    else:
        raise ValueError(f"unknown excerpt assignment policy: {assignment_policy}")

    sample_rates_hz = {excerpt.sample_rate_hz for excerpt in excerpts}
    if len(sample_rates_hz) > 1:
        raise ValueError(
            f"the selected excerpts disagree on sample rate: {sample_rates_hz}"
        )

    achieved_coherence = compute_maximum_off_diagonal_coherence(
        compute_pairwise_excerpt_coherence_matrix(
            [excerpt.samples for excerpt in excerpts]
        )
    )
    if achieved_coherence > maximum_permitted_excerpt_coherence:
        raise ValueError(
            f"the selected excerpts share waveform content at a coherence of "
            f"{achieved_coherence:.3f}, above the permitted "
            f"{maximum_permitted_excerpt_coherence:.3f}; every receiver pair's "
            f"cross-correlation would carry an artefact indistinguishable from a "
            f"real source"
        )
    return excerpts
