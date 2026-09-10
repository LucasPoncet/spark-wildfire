"""Reads one forward run back. Moves bytes, transforms nothing.

Every consumer of a run — the estimator, the plotters, the application —
loads it through here, so none of them imports a script and none of them
hard-codes a filename. The layout this reads is fixed by
`simulation_run_writer.py`.

Signals are memory-mapped rather than read: a run is hundreds of megabytes
and the application must not block on opening one. Slice the array to pull
only the observations you need into memory.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.array_types import Float32Array, Float64Array
from src.utils.io.simulation_run_writer import (
    CONFIG_FILENAME,
    GROUND_TRUTH_FILENAME,
    METADATA_FILENAME,
    RECEIVER_POSITIONS_FILENAME,
    RECEIVER_SIGNALS_FILENAME,
    JsonDocument,
)


@dataclass(frozen=True)
class SimulationRun:
    """One saved forward run, with its signals still on disk.

    Attributes:
        run_directory: Directory the run was read from.
        config: The resolved forward configuration as written.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        ground_truth: One record per observation.
        metadata: Timestep, observation cadence, sample rate, receiver count.
    """

    run_directory: Path
    config: dict[str, Any]
    receiver_positions_xy_m: Float64Array
    ground_truth: list[dict[str, Any]]
    metadata: dict[str, Any]

    @property
    def run_id(self) -> str:
        """Identifier of this run, matching what `list_simulation_runs` returns.

        Runs are nested by experiment, so the identifier is `<experiment>/<stamp>`
        whenever the metadata names one. A run written before that nesting
        existed keeps its bare stamp.
        """
        experiment_name = self.metadata.get("experiment_name")
        if isinstance(experiment_name, str) and experiment_name:
            return f"{experiment_name}/{self.run_directory.name}"
        return self.run_directory.name

    @property
    def experiment_name(self) -> str:
        """Name of the scene this run came from, or the directory name."""
        experiment_name = self.metadata.get("experiment_name")
        if isinstance(experiment_name, str) and experiment_name:
            return experiment_name
        return self.run_directory.name

    @property
    def receiver_signals(self) -> Float32Array:
        """Signals as a read-only memory map of shape (n_obs, n_rx, n_samples)."""
        return cast(
            Float32Array,
            np.load(self.run_directory / RECEIVER_SIGNALS_FILENAME, mmap_mode="r"),
        )

    @property
    def observation_times_s(self) -> Float64Array:
        """Simulated time of each observation, in seconds."""
        return np.array(
            [float(record["simulation_time_s"]) for record in self.ground_truth],
            dtype=np.float64,
        )

    @property
    def receiver_count(self) -> int:
        """Number of receivers the run was rendered to."""
        return int(self.receiver_positions_xy_m.shape[0])

    @property
    def observation_count(self) -> int:
        """Number of acoustic observations the run holds."""
        return len(self.ground_truth)


def read_json_document(path: Path) -> JsonDocument:
    """Read one JSON document.

    Args:
        path: Path to the file.

    Returns:
        The parsed document.

    Raises:
        FileNotFoundError: If the file does not exist, naming it.
    """
    if not path.is_file():
        raise FileNotFoundError(f"simulation run file not found: {path}")
    return cast(JsonDocument, json.loads(path.read_text(encoding="utf-8")))


def read_configuration_document(path: Path) -> dict[str, Any]:
    """Read one JSON document known to be a mapping.

    Args:
        path: Path to the file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the document is not a mapping.
    """
    document = read_json_document(path)
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return document


def read_record_list(path: Path) -> list[dict[str, Any]]:
    """Read one JSON document known to be a list of mappings.

    Args:
        path: Path to the file.

    Returns:
        The parsed records.

    Raises:
        ValueError: If the document is not a list.
    """
    document = read_json_document(path)
    if not isinstance(document, list):
        raise ValueError(f"expected a JSON array in {path}")
    return document


def read_simulation_run(run_directory: Path) -> SimulationRun:
    """Read one run directory, leaving the signal array on disk.

    Args:
        run_directory: Directory holding the five run files.

    Returns:
        The run, with signals reachable through `receiver_signals`.

    Raises:
        FileNotFoundError: If the directory or any of its files is missing.
    """
    if not run_directory.is_dir():
        raise FileNotFoundError(f"simulation run directory not found: {run_directory}")

    signals_path = run_directory / RECEIVER_SIGNALS_FILENAME
    if not signals_path.is_file():
        raise FileNotFoundError(f"simulation run file not found: {signals_path}")

    positions_path = run_directory / RECEIVER_POSITIONS_FILENAME
    if not positions_path.is_file():
        raise FileNotFoundError(f"simulation run file not found: {positions_path}")

    return SimulationRun(
        run_directory=run_directory,
        config=read_configuration_document(run_directory / CONFIG_FILENAME),
        receiver_positions_xy_m=np.asarray(np.load(positions_path), dtype=np.float64),
        ground_truth=read_record_list(run_directory / GROUND_TRUTH_FILENAME),
        metadata=read_configuration_document(run_directory / METADATA_FILENAME),
    )


def list_simulation_runs(root: Path) -> list[str]:
    """List the run identifiers under a root, newest first.

    Runs are nested one level by experiment — `results/simulations/e3/<stamp>/`
    — so an identifier is `e3/<stamp>` and `root / run_id` still opens it. A
    run written before that nesting existed sits directly under the root and is
    listed by its bare stamp, so an older tree keeps working.

    A directory counts as a run when it holds a metadata sidecar, so a
    half-written or unrelated directory is skipped rather than crashing the
    caller. Identifiers end in a UTC timestamp, so sorting on that segment
    sorts them newest first.

    Args:
        root: Directory holding run directories, possibly nested by experiment.

    Returns:
        Run identifiers as `/`-separated relative paths, newest first, empty
        when the root does not exist.
    """
    if not root.is_dir():
        return []
    run_directories = [
        candidate.parent for candidate in root.glob(f"*/{METADATA_FILENAME}")
    ] + [candidate.parent for candidate in root.glob(f"*/*/{METADATA_FILENAME}")]
    return sorted(
        (directory.relative_to(root).as_posix() for directory in run_directories),
        key=lambda run_id: (run_id.rsplit("/", 1)[-1], run_id),
        reverse=True,
    )
