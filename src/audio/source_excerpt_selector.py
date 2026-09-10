from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.audio.excerpt_coherence import (
    compute_maximum_off_diagonal_coherence,
    compute_pairwise_excerpt_coherence_matrix,
)
from src.utils.array_types import Float64Array
from src.utils.io.audio_file_reader import read_audio_file

EXPLICIT_ASSIGNMENT: str = "explicit"
DISTINCT_PROVENANCE_ASSIGNMENT: str = "distinct_provenance"
MINIMUM_COHERENCE_ASSIGNMENT: str = "minimum_coherence"


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
) -> list[SourceExcerpt]:
    """Assigns one recording excerpt to each source and checks they are distinct.

    Args:
        recording_paths: Candidate recordings.
        source_count: Number of sources to feed.
        clip_duration_s: Excerpt length in seconds.
        assignment_policy: `explicit`, `distinct_provenance` or `minimum_coherence`.
        maximum_permitted_excerpt_coherence: Largest mutual coherence a run accepts.
        selection_seed: Seed making a shuffled assignment reproducible.
        as_mono: Whether to average the channels down to one.

    Returns:
        One excerpt per source.

    Raises:
        ValueError: If the policy is unknown, the pool is too small, the sample
            rates disagree, or the excerpts are more mutually coherent than
            permitted.
    """
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if len(recording_paths) < source_count:
        raise ValueError(
            f"the pool holds {len(recording_paths)} recordings, fewer than the "
            f"{source_count} sources requested"
        )

    if assignment_policy == EXPLICIT_ASSIGNMENT:
        selected_paths = recording_paths[:source_count]
    elif assignment_policy == DISTINCT_PROVENANCE_ASSIGNMENT:
        order = np.random.default_rng(selection_seed).permutation(len(recording_paths))
        selected_paths = [recording_paths[index] for index in order[:source_count]]
    elif assignment_policy == MINIMUM_COHERENCE_ASSIGNMENT:
        pool = [
            read_leading_excerpt(path, clip_duration_s, as_mono)
            for path in recording_paths
        ]
        indices = choose_least_coherent_indices(
            compute_pairwise_excerpt_coherence_matrix(
                [excerpt.samples for excerpt in pool]
            ),
            source_count,
        )
        selected_paths = [pool[index].recording_path for index in indices]
    else:
        raise ValueError(f"unknown excerpt assignment policy: {assignment_policy}")

    excerpts = [
        read_leading_excerpt(path, clip_duration_s, as_mono) for path in selected_paths
    ]
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
