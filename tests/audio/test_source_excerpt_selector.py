from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio.source_excerpt_selector import (
    DISTINCT_PROVENANCE_ASSIGNMENT,
    DISTINCT_WINDOWS_ASSIGNMENT,
    EXPLICIT_ASSIGNMENT,
    MINIMUM_COHERENCE_ASSIGNMENT,
    choose_least_coherent_indices,
    enumerate_pool_windows,
    list_pool_recordings,
    read_excerpt_at_offset,
    read_leading_excerpt,
    select_source_excerpts,
)

CLIP_DURATION_S: float = 0.5
PERMISSIVE_COHERENCE: float = 1.0


def write_pool(
    directory: Path, sample_rate_hz: int, recording_count: int
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(0)
    paths = []
    for index in range(recording_count):
        path = directory / f"recording_{index:02d}.wav"
        sf.write(path, generator.standard_normal(2 * sample_rate_hz), sample_rate_hz)
        paths.append(path)
    return paths


def test_the_pool_defaults_to_every_wav_file_sorted(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    written = write_pool(tmp_path / "pool", sample_rate_hz, 3)
    assert list_pool_recordings(tmp_path / "pool", ()) == sorted(written)


def test_naming_recordings_fixes_their_order(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    write_pool(tmp_path / "pool", sample_rate_hz, 3)
    paths = list_pool_recordings(
        tmp_path / "pool", ("recording_02.wav", "recording_00.wav")
    )
    assert [path.name for path in paths] == ["recording_02.wav", "recording_00.wav"]


def test_a_missing_pool_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="recording directory not found"):
        list_pool_recordings(tmp_path / "absent", ())


def test_an_excerpt_is_cut_to_the_requested_duration(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    path = write_pool(tmp_path / "pool", sample_rate_hz, 1)[0]
    excerpt = read_leading_excerpt(path, CLIP_DURATION_S, True)
    assert excerpt.samples.size == round(CLIP_DURATION_S * sample_rate_hz)
    assert excerpt.sample_rate_hz == sample_rate_hz
    assert excerpt.recording_path == path


def test_a_recording_shorter_than_the_clip_is_rejected(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    path = write_pool(tmp_path / "pool", sample_rate_hz, 1)[0]
    with pytest.raises(ValueError, match="shorter than the requested"):
        read_leading_excerpt(path, 10.0, True)


def test_distinct_provenance_never_reuses_a_recording(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    paths = list_pool_recordings(
        write_pool(tmp_path / "pool", sample_rate_hz, 5)[0].parent, ()
    )
    excerpts = select_source_excerpts(
        paths,
        3,
        CLIP_DURATION_S,
        DISTINCT_PROVENANCE_ASSIGNMENT,
        PERMISSIVE_COHERENCE,
        0,
        True,
    )
    assert len({excerpt.recording_path for excerpt in excerpts}) == 3


def test_a_pool_smaller_than_the_source_count_is_rejected(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    paths = list_pool_recordings(
        write_pool(tmp_path / "pool", sample_rate_hz, 2)[0].parent, ()
    )
    with pytest.raises(ValueError, match="fewer than the 3 sources requested"):
        select_source_excerpts(
            paths,
            3,
            CLIP_DURATION_S,
            EXPLICIT_ASSIGNMENT,
            PERMISSIVE_COHERENCE,
            0,
            True,
        )


def test_excerpts_above_the_coherence_limit_are_rejected(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    directory = tmp_path / "pool"
    directory.mkdir(parents=True)
    samples = np.random.default_rng(7).standard_normal(2 * sample_rate_hz)
    for index in range(2):
        sf.write(directory / f"copy_{index}.wav", samples, sample_rate_hz)
    with pytest.raises(ValueError, match="share waveform content"):
        select_source_excerpts(
            list_pool_recordings(directory, ()),
            2,
            CLIP_DURATION_S,
            EXPLICIT_ASSIGNMENT,
            0.15,
            0,
            True,
        )


def test_an_unknown_policy_is_rejected(tmp_path: Path, sample_rate_hz: int) -> None:
    paths = list_pool_recordings(
        write_pool(tmp_path / "pool", sample_rate_hz, 3)[0].parent, ()
    )
    with pytest.raises(ValueError, match="unknown excerpt assignment policy"):
        select_source_excerpts(
            paths, 2, CLIP_DURATION_S, "nonsense", PERMISSIVE_COHERENCE, 0, True
        )


def test_the_minimum_coherence_policy_avoids_the_duplicated_pair(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    directory = tmp_path / "pool"
    directory.mkdir(parents=True)
    generator = np.random.default_rng(11)
    duplicated = generator.standard_normal(2 * sample_rate_hz)
    sf.write(directory / "a_copy_one.wav", duplicated, sample_rate_hz)
    sf.write(directory / "b_copy_two.wav", duplicated, sample_rate_hz)
    sf.write(
        directory / "c_independent.wav",
        generator.standard_normal(2 * sample_rate_hz),
        sample_rate_hz,
    )
    excerpts = select_source_excerpts(
        list_pool_recordings(directory, ()),
        2,
        CLIP_DURATION_S,
        MINIMUM_COHERENCE_ASSIGNMENT,
        0.15,
        0,
        True,
    )
    names = {excerpt.recording_path.name for excerpt in excerpts}
    assert "c_independent.wav" in names
    assert names != {"a_copy_one.wav", "b_copy_two.wav"}


def test_the_greedy_subset_picks_the_least_coherent_pair() -> None:
    coherence_matrix = np.array(
        [
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.1],
            [0.8, 0.1, 1.0],
        ]
    )
    assert sorted(choose_least_coherent_indices(coherence_matrix, 2)) == [1, 2]


def test_a_subset_larger_than_the_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="fewer than the 3 sources requested"):
        choose_least_coherent_indices(np.eye(2), 3)


def test_an_offset_excerpt_starts_where_it_was_asked_to(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    directory = tmp_path / "pool"
    directory.mkdir(parents=True)
    samples = 0.2 * np.random.default_rng(21).standard_normal(2 * sample_rate_hz)
    path = directory / "one.wav"
    sf.write(path, samples, sample_rate_hz)

    offset = sample_rate_hz
    excerpt = read_excerpt_at_offset(path, offset, CLIP_DURATION_S, True)
    assert excerpt.start_sample_index == offset
    assert excerpt.samples.size == round(CLIP_DURATION_S * sample_rate_hz)
    assert np.allclose(
        excerpt.samples, samples[offset : offset + excerpt.samples.size], atol=1e-4
    )


def test_an_offset_past_the_end_of_a_recording_is_rejected(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    path = write_pool(tmp_path / "pool", sample_rate_hz, 1)[0]
    with pytest.raises(ValueError, match="ends before"):
        read_excerpt_at_offset(path, 2 * sample_rate_hz, CLIP_DURATION_S, True)


def test_the_pool_is_cut_into_non_overlapping_windows(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    paths = write_pool(tmp_path / "pool", sample_rate_hz, 2)
    windows = enumerate_pool_windows(paths, CLIP_DURATION_S, True)
    clip_sample_count = round(CLIP_DURATION_S * sample_rate_hz)
    assert len(windows) == 2 * (2 * sample_rate_hz // clip_sample_count)
    for path in paths:
        offsets = [
            window.start_sample_index
            for window in windows
            if window.recording_path == path
        ]
        assert offsets == sorted(offsets)
        assert all(
            later - earlier >= clip_sample_count for earlier, later in pairwise(offsets)
        )


def test_distinct_windows_can_feed_more_sources_than_there_are_files(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    """Why the policy exists at all.

    A pool of two recordings offers two excerpts at full length, but many more
    once each is cut into non-overlapping windows, which is what lets a scene
    sound more sources than the pool holds files.
    """
    paths = write_pool(tmp_path / "pool", sample_rate_hz, 2)
    excerpts = select_source_excerpts(
        paths,
        5,
        CLIP_DURATION_S,
        DISTINCT_WINDOWS_ASSIGNMENT,
        PERMISSIVE_COHERENCE,
        0,
        True,
    )
    assert len(excerpts) == 5
    assert len({(e.recording_path, e.start_sample_index) for e in excerpts}) == 5


def test_explicit_offsets_name_a_window_set_without_searching(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    """How a scene records a window set the audit already found.

    Naming one file three times with three offsets must give three different
    excerpts, so a scene can sound more sources than the pool holds files
    without repeating the coherence search on every run.
    """
    directory = tmp_path / "pool"
    directory.mkdir(parents=True)
    samples = 0.2 * np.random.default_rng(31).standard_normal(3 * sample_rate_hz)
    path = directory / "one.wav"
    sf.write(path, samples, sample_rate_hz)

    excerpts = select_source_excerpts(
        [path, path, path],
        3,
        CLIP_DURATION_S,
        EXPLICIT_ASSIGNMENT,
        PERMISSIVE_COHERENCE,
        0,
        True,
        (0.0, 1.0, 2.0),
    )
    assert [excerpt.start_sample_index for excerpt in excerpts] == [
        0,
        sample_rate_hz,
        2 * sample_rate_hz,
    ]
    assert not np.allclose(excerpts[0].samples, excerpts[1].samples)


def test_naming_no_offsets_still_reads_from_the_start(
    tmp_path: Path, sample_rate_hz: int
) -> None:
    paths = write_pool(tmp_path / "pool", sample_rate_hz, 2)
    excerpts = select_source_excerpts(
        paths, 2, CLIP_DURATION_S, EXPLICIT_ASSIGNMENT, PERMISSIVE_COHERENCE, 0, True
    )
    assert [excerpt.start_sample_index for excerpt in excerpts] == [0, 0]


def test_too_few_named_offsets_is_rejected(tmp_path: Path, sample_rate_hz: int) -> None:
    paths = write_pool(tmp_path / "pool", sample_rate_hz, 3)
    with pytest.raises(ValueError, match="fewer than the 3 sources requested"):
        select_source_excerpts(
            paths,
            3,
            CLIP_DURATION_S,
            EXPLICIT_ASSIGNMENT,
            PERMISSIVE_COHERENCE,
            0,
            True,
            (0.0, 0.5),
        )
