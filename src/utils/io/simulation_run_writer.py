"""Writes one forward run to disk. Moves bytes, transforms nothing.

The run directory is the whole contract between the forward side, which
produces scenes, and the inverse side, which consumes them. Its layout is
fixed here and read back by `simulation_run_reader.py`; changing one without
the other breaks every consumer silently.

    config.json                     resolved ForwardSimulationConfiguration
    receiver_positions_xy_m.npy     float64, (n_receivers, 2)
    receiver_signals.npy            float32, (n_observations, n_receivers, n_samples)
    ground_truth.json               list of n_observations records, each carrying
                                    simulation_time_s, component_count,
                                    component_centroids_xy_m, front_cell_count,
                                    burning_cell_count, source_count
    metadata.json                   time_step_s, observation_interval_s,
                                    observation_count, sample_rate_hz, receiver_count

Signals are stored as float32 deliberately: at a 20 dB signal-to-noise ratio
no estimator can use float64 precision, and it halves every run on disk.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.array_types import Float32Array, Float64Array

CONFIG_FILENAME: str = "config.json"
RECEIVER_POSITIONS_FILENAME: str = "receiver_positions_xy_m.npy"
RECEIVER_SIGNALS_FILENAME: str = "receiver_signals.npy"
GROUND_TRUTH_FILENAME: str = "ground_truth.json"
METADATA_FILENAME: str = "metadata.json"

type JsonDocument = dict[str, Any] | list[dict[str, Any]]


def write_json_document(document: JsonDocument, path: Path) -> None:
    """Write one indented JSON document with a trailing newline.

    Args:
        document: A mapping, or a list of mappings.
        path: Destination file.
    """
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def write_simulation_run(
    run_directory: Path,
    config: dict[str, Any],
    receiver_positions_xy_m: Float64Array,
    receiver_signals: Float32Array,
    ground_truth: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Write one whole run, creating the directory and its parents.

    Args:
        run_directory: Directory to write into.
        config: The resolved forward configuration, already a mapping.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        receiver_signals: Float32 array of shape
            (n_observations, n_receivers, n_samples).
        ground_truth: One record per observation.
        metadata: The sidecar values named in the module docstring.

    Returns:
        The directory written to.

    Raises:
        ValueError: If the signal array is not three-dimensional, or if its
            observation count disagrees with the ground truth.
    """
    if receiver_signals.ndim != 3:
        raise ValueError(
            "receiver signals must have shape "
            f"(n_observations, n_receivers, n_samples), got {receiver_signals.shape}"
        )
    if receiver_signals.shape[0] != len(ground_truth):
        raise ValueError(
            f"{receiver_signals.shape[0]} observations rendered but "
            f"{len(ground_truth)} ground truth records supplied"
        )

    run_directory.mkdir(parents=True, exist_ok=True)
    write_json_document(config, run_directory / CONFIG_FILENAME)
    np.save(
        run_directory / RECEIVER_POSITIONS_FILENAME,
        np.asarray(receiver_positions_xy_m, dtype=np.float64),
    )
    np.save(
        run_directory / RECEIVER_SIGNALS_FILENAME,
        np.asarray(receiver_signals, dtype=np.float32),
    )
    write_json_document(ground_truth, run_directory / GROUND_TRUTH_FILENAME)
    write_json_document(metadata, run_directory / METADATA_FILENAME)
    return run_directory
