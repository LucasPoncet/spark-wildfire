"""Audits the recording pool before any localization is attempted.

Usage:
    uv run python scripts/run_source_excerpt_audit.py
    uv run python scripts/run_source_excerpt_audit.py --configs other_dir

Stage 0 of the multi-source pipeline. It measures each recording's codec cutoff,
resolves the two band lists the estimator needs, selects one excerpt per source,
computes the mutual coherence matrix and fails loudly when two excerpts share
waveform content: every receiver pair's cross-correlation would then carry a
peak at a lag no source occupies, and nothing downstream can tell that artefact
from a real fire.

Run this before anything else.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.audio.codec_bandwidth_detector import detect_codec_cutoff_hz
from src.audio.excerpt_coherence import (
    compute_maximum_off_diagonal_coherence,
    compute_pairwise_excerpt_coherence_matrix,
)
from src.audio.octave_band_filter import (
    build_fractional_octave_centre_frequencies_hz,
)
from src.audio.source_excerpt_selector import (
    list_pool_recordings,
    read_leading_excerpt,
    select_source_excerpts,
)
from src.audio.usable_band_selector import select_usable_band_centres_hz
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    SimulationConfiguration,
    load_simulation_configuration,
)
from src.utils.array_types import Float64Array
from src.utils.io.metrics_writer import write_metrics_document

AUDIT_METRICS_FILENAME: str = "source_excerpt_audit.json"


def parse_arguments() -> argparse.Namespace:
    """Parses the only argument this script takes.

    Returns:
        The parsed arguments, carrying the configuration directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    return parser.parse_args()


def measure_pool_cutoffs_hz(
    recording_paths: list[Path], configuration: SimulationConfiguration
) -> dict[str, float]:
    """Measures the codec cutoff of every recording in the pool.

    Args:
        recording_paths: Candidate recordings.
        configuration: The whole run configuration.

    Returns:
        Cutoff in hertz, keyed by file name.
    """
    bands = configuration.localization.multi_source.bands
    cutoffs_hz: dict[str, float] = {}
    for path in recording_paths:
        excerpt = read_leading_excerpt(
            path,
            configuration.data.segmentation.clip_duration_s,
            configuration.data.recording.as_mono,
        )
        cutoffs_hz[path.name] = detect_codec_cutoff_hz(
            excerpt.samples,
            excerpt.sample_rate_hz,
            bands.codec_detection_window_sample_count,
            bands.codec_detection_overlap_fraction,
            bands.codec_detection_floor_drop_db,
        )
    return cutoffs_hz


def resolve_band_lists(
    configuration: SimulationConfiguration, codec_cutoff_hz: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Resolves the level-stage band centres against a measured cutoff.

    Args:
        configuration: The whole run configuration.
        codec_cutoff_hz: Lowest cutoff measured across the pool, in hertz.

    Returns:
        `(candidate_centres_hz, usable_centres_hz)`.
    """
    bands = configuration.localization.multi_source.bands
    candidate_centres_hz = build_fractional_octave_centre_frequencies_hz(
        bands.lowest_centre_hz, bands.highest_centre_hz, bands.fraction_denominator
    )
    return candidate_centres_hz, select_usable_band_centres_hz(
        candidate_centres_hz,
        codec_cutoff_hz,
        bands.codec_cutoff_margin,
        bands.fraction_denominator,
    )


def format_coherence_matrix(
    coherence_matrix: Float64Array, labels: list[str]
) -> list[str]:
    """Formats a coherence matrix as an aligned table.

    Args:
        coherence_matrix: Square matrix of mutual coherences.
        labels: One short label per excerpt.

    Returns:
        Lines ready to print.
    """
    lines = ["  " + " ".join(f"{label:>10}" for label in ["", *labels])]
    for label, row in zip(labels, np.asarray(coherence_matrix), strict=True):
        lines.append(
            "  " + f"{label:>10} " + " ".join(f"{float(value):>10.3f}" for value in row)
        )
    return lines


def print_pool_report(
    recording_paths: list[Path],
    cutoffs_hz: dict[str, float],
    candidate_centres_hz: tuple[float, ...],
    usable_centres_hz: tuple[float, ...],
    configuration: SimulationConfiguration,
) -> None:
    """Prints the pool bandwidth findings and the resolved band lists.

    Args:
        recording_paths: Candidate recordings.
        cutoffs_hz: Cutoff per file name.
        candidate_centres_hz: Band centres offered by the configuration.
        usable_centres_hz: Band centres that survive the codec rule.
        configuration: The whole run configuration.
    """
    correlation = configuration.localization.multi_source.correlation
    print(f"configs        : {configuration.configuration_directory}")
    print(f"pool           : {configuration.data.excerpts.recording_directory}")
    print(f"recordings     : {len(recording_paths)}")
    print()
    print("codec cutoff per recording")
    for path in recording_paths:
        print(f"  {path.name:>28}  {cutoffs_hz[path.name]:>9.0f} Hz")
    print()
    print(
        f"correlation band: {correlation.lowest_frequency_hz:.0f} - "
        f"{correlation.highest_frequency_hz:.0f} Hz, one continuous passband"
    )
    print(
        f"level bands     : {len(usable_centres_hz)} of "
        f"{len(candidate_centres_hz)} candidates survive the codec rule"
    )
    print(f"  usable        : {[round(centre) for centre in usable_centres_hz]}")
    dropped = [
        round(centre)
        for centre in candidate_centres_hz
        if centre not in usable_centres_hz
    ]
    print(f"  dropped       : {dropped}")
    print()


def main() -> None:
    """Audits the pool and writes the audit document, or fails loudly."""
    arguments = parse_arguments()
    configuration = load_simulation_configuration(arguments.configs)
    excerpts_configuration = configuration.data.excerpts

    recording_paths = list_pool_recordings(
        excerpts_configuration.recording_directory,
        excerpts_configuration.recording_filenames,
    )
    cutoffs_hz = measure_pool_cutoffs_hz(recording_paths, configuration)
    lowest_cutoff_hz = min(cutoffs_hz.values())
    candidate_centres_hz, usable_centres_hz = resolve_band_lists(
        configuration, lowest_cutoff_hz
    )
    print_pool_report(
        recording_paths,
        cutoffs_hz,
        candidate_centres_hz,
        usable_centres_hz,
        configuration,
    )

    pool_excerpts = [
        read_leading_excerpt(
            path,
            configuration.data.segmentation.clip_duration_s,
            configuration.data.recording.as_mono,
        )
        for path in recording_paths
    ]
    pool_coherence = compute_pairwise_excerpt_coherence_matrix(
        [excerpt.samples for excerpt in pool_excerpts]
    )
    print("pool mutual coherence")
    for line in format_coherence_matrix(
        pool_coherence, [path.stem[-6:] for path in recording_paths]
    ):
        print(line)
    worst_pool_coherence = compute_maximum_off_diagonal_coherence(pool_coherence)
    print(
        f"  -> worst off-diagonal {worst_pool_coherence:.3f} | permitted "
        f"{excerpts_configuration.maximum_permitted_excerpt_coherence:.3f}"
    )
    print()

    source_count = len(configuration.geometry.concurrent_sources)
    selection_error = ""
    selected_names: list[str] = []
    selected_coherence = np.eye(source_count, dtype=np.float64)
    try:
        selected = select_source_excerpts(
            recording_paths,
            source_count,
            configuration.data.segmentation.clip_duration_s,
            excerpts_configuration.assignment_policy,
            excerpts_configuration.maximum_permitted_excerpt_coherence,
            excerpts_configuration.selection_seed,
            configuration.data.recording.as_mono,
        )
        selected_names = [excerpt.recording_path.name for excerpt in selected]
        selected_coherence = compute_pairwise_excerpt_coherence_matrix(
            [excerpt.samples for excerpt in selected]
        )
        print(f"selected excerpts ({excerpts_configuration.assignment_policy})")
        for source_index, name in enumerate(selected_names):
            print(f"  source {source_index}: {name}")
        print(
            f"  -> worst off-diagonal "
            f"{compute_maximum_off_diagonal_coherence(selected_coherence):.3f}"
        )
    except ValueError as error:
        selection_error = str(error)
        print(f"SELECTION FAILED: {selection_error}")
    print()

    metrics_path = write_metrics_document(
        {
            "configuration_directory": str(configuration.configuration_directory),
            "recording_directory": str(excerpts_configuration.recording_directory),
            "recording_names": [path.name for path in recording_paths],
            "codec_cutoff_hz": cutoffs_hz,
            "lowest_codec_cutoff_hz": lowest_cutoff_hz,
            "correlation_band_hz": [
                configuration.localization.multi_source.correlation.lowest_frequency_hz,
                configuration.localization.multi_source.correlation.highest_frequency_hz,
            ],
            "candidate_band_centres_hz": list(candidate_centres_hz),
            "usable_band_centres_hz": list(usable_centres_hz),
            "pool_coherence_matrix": pool_coherence,
            "pool_maximum_off_diagonal_coherence": worst_pool_coherence,
            "maximum_permitted_excerpt_coherence": (
                excerpts_configuration.maximum_permitted_excerpt_coherence
            ),
            "assignment_policy": excerpts_configuration.assignment_policy,
            "selected_recording_names": selected_names,
            "selected_coherence_matrix": selected_coherence,
            "selection_error": selection_error,
        },
        configuration.data.output.metrics_directory / AUDIT_METRICS_FILENAME,
    )
    print(f"audit written to {metrics_path}")
    if selection_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
