"""Loading and derived quantities the panels share. No plotting, no physics.

Everything here is either a call into `src/` or arithmetic on what came back.
If a function in this package starts deciding something scientific, it belongs
in `src/` instead.
"""

from pathlib import Path

import numpy as np
import streamlit as st

from src.config.experiment_configuration import format_scene_label
from src.config.simulation_configuration import ForwardSimulationConfiguration
from src.config.simulation_context import SimulationContext
from src.config.simulation_context_factory import build_simulation_context
from src.utils.array_types import Float64Array
from src.utils.io.simulation_run_reader import (
    SimulationRun,
    list_simulation_runs,
    read_simulation_run,
)

RUN_ROOT: Path = Path("results/simulations")
CONFIGURATION_ROOT: Path = Path("configs")
LADDER_DIRECTORIES: tuple[Path, ...] = (
    CONFIGURATION_ROOT / "e1",
    CONFIGURATION_ROOT / "e2",
    CONFIGURATION_ROOT / "e3",
    CONFIGURATION_ROOT / "e4",
)
LEVEL_FLOOR: float = 1e-20
DECIBELS_PER_POWER_DECADE: float = 10.0


def available_run_ids() -> list[str]:
    """Run identifiers under the run root, newest first.

    Returns:
        Identifiers, empty when nothing has been rendered yet.
    """
    return list_simulation_runs(RUN_ROOT)


@st.cache_resource(show_spinner=False)
def load_run(run_id: str) -> SimulationRun:
    """Read one run, memoised so switching panels does not re-read it.

    Signals stay memory-mapped, so this is cheap even for a large run.

    Args:
        run_id: Identifier of the run directory.

    Returns:
        The run.
    """
    return read_simulation_run(RUN_ROOT / run_id)


@st.cache_resource(show_spinner=False)
def load_context(configuration_directory: str) -> SimulationContext:
    """Build one scene's live objects, memoised across reruns.

    Args:
        configuration_directory: Directory naming the scene.

    Returns:
        The simulation context.
    """
    from src.config.simulation_configuration import (
        load_forward_simulation_configuration,
    )

    return build_simulation_context(
        load_forward_simulation_configuration(Path(configuration_directory))
    )


@st.cache_data(show_spinner=False)
def load_configuration(configuration_directory: str) -> ForwardSimulationConfiguration:
    """Read one scene's configuration, memoised across reruns.

    Args:
        configuration_directory: Directory naming the scene.

    Returns:
        The configuration.
    """
    from src.config.simulation_configuration import (
        load_forward_simulation_configuration,
    )

    return load_forward_simulation_configuration(Path(configuration_directory))


def compute_receiver_levels_db(run: SimulationRun) -> Float64Array:
    """Broadband level at each receiver for each observation of a run.

    Args:
        run: The saved run.

    Returns:
        Float64 array of shape (n_observations, n_receivers), in decibels.
    """
    signals = np.asarray(run.receiver_signals, dtype=np.float64)
    mean_power = np.mean(signals**2, axis=2)
    return np.asarray(
        DECIBELS_PER_POWER_DECADE * np.log10(mean_power + LEVEL_FLOOR),
        dtype=np.float64,
    )


def read_ground_truth_series(run: SimulationRun, field: str) -> Float64Array:
    """Pull one scalar field out of every ground-truth record.

    Args:
        run: The saved run.
        field: Record key to read.

    Returns:
        Float64 array of shape (n_observations,).
    """
    return np.array(
        [float(record[field]) for record in run.ground_truth], dtype=np.float64
    )


def available_ladder_directories() -> list[Path]:
    """Ladder scene directories that exist on disk.

    Returns:
        The directories, in ladder order.
    """
    return [directory for directory in LADDER_DIRECTORIES if directory.is_dir()]


def ladder_scene_options() -> dict[str, Path]:
    """Ladder scenes as picker labels mapped back to their directories.

    Labels carry the scene's own title, so a reader picking "e3" sees what E3
    is without opening a file.

    Returns:
        Mapping of label to configuration directory, in ladder order.
    """
    options: dict[str, Path] = {}
    for directory in available_ladder_directories():
        experiment = load_configuration(str(directory)).experiment
        options[format_scene_label(experiment.name, experiment.title)] = directory
    return options
