"""Runs a fire simulation and renders receiver signals at every timestep.

Usage:
    uv run python scripts/run_acoustic_rendering.py
    uv run python scripts/run_acoustic_rendering.py --configs other_dir

This is the handoff to the inverse team: it writes receiver positions, a
`(timestep, receiver, sample)` signal array, per-timestep ground truth, the
resolved configuration and a metadata sidecar into
`results/simulation_runs/<timestamp>/`. Zero domain logic lives here.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config.fire_simulation_configuration import (
    ALL_BURNING_EMISSION,
    FRONT_ONLY_EMISSION,
)
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    ForwardSimulationConfiguration,
    load_forward_simulation_configuration,
)
from src.spark.acoustic.burning_cell_source_model import (
    BurningCellSources,
    compute_fire_front_mask,
    extract_burning_cell_sources,
    extract_fire_front_sources,
    generate_source_signal_for_burning_cell,
    identify_connected_front_components,
)
from src.spark.acoustic.free_field_propagation import render_receiver_signals
from src.spark.acoustic.receiver_placement import place_receivers_from_configuration
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.fire_state import FireState
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_engine import RateOfSpreadEngine
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.terrain.square_grid_mesh import SquareGridMesh, SquareGridMeshConfig
from src.utils.array_types import Float64Array, Int64Array

OUTPUT_ROOT: Path = Path("results/simulation_runs")
PROGRESS_REPORT_INTERVAL_STEPS: int = 50
FUEL_PRESETS = {"pine_needle_litter": FuelProperties.pine_needle_litter}


def parse_arguments() -> argparse.Namespace:
    """Parses the only argument this script takes.

    Returns:
        The parsed arguments, carrying the configuration directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    return parser.parse_args()


def build_fuel(fuel_preset_name: str) -> FuelProperties:
    """Look up a fuel preset by the name the configuration carries.

    Args:
        fuel_preset_name: Key into the supported preset table.

    Returns:
        The fuel bed.

    Raises:
        ValueError: If the preset name is not supported.
    """
    if fuel_preset_name not in FUEL_PRESETS:
        supported = ", ".join(sorted(FUEL_PRESETS))
        raise ValueError(
            f"unknown fuel preset {fuel_preset_name!r}; supported presets: {supported}"
        )
    return FUEL_PRESETS[fuel_preset_name]()


def compute_ignition_cell_index(
    mesh: SquareGridMesh, x_fraction: float, y_fraction: float
) -> int:
    """Turn fractional grid coordinates into a row-major cell index.

    Args:
        mesh: The grid the fire runs on.
        x_fraction: Position along x, 0.0 at the origin corner, 1.0 at the far edge.
        y_fraction: Position along y, same convention.

    Returns:
        Index into the mesh's cell arrays.
    """
    ignition_x = int(x_fraction * (mesh.n_x - 1))
    ignition_y = int(y_fraction * (mesh.n_y - 1))
    return ignition_y * mesh.n_x + ignition_x


def extract_sources(
    emission_model: str,
    fire_state: FireState,
    mesh: SquareGridMesh,
    fuel: FuelProperties,
) -> BurningCellSources:
    """Pick the emission model the configuration names.

    Args:
        emission_model: Either "front_only" or "all_burning".
        fire_state: State to read burning and ignited cells from.
        mesh: The grid, supplying positions and connectivity.
        fuel: The fuel bed, supplying load and residence time.

    Returns:
        The radiating cells as sources.

    Raises:
        ValueError: If the emission model is not recognised.
    """
    if emission_model == FRONT_ONLY_EMISSION:
        return extract_fire_front_sources(
            fire_state,
            mesh.cell_positions_xyz,
            mesh.neighbor_indices,
            fuel.fuel_load_kg_per_m2,
            fuel.residence_time_s,
        )
    if emission_model == ALL_BURNING_EMISSION:
        return extract_burning_cell_sources(
            fire_state,
            mesh.cell_positions_xyz,
            fuel.fuel_load_kg_per_m2,
            fuel.residence_time_s,
        )
    raise ValueError(f"unknown emission model: {emission_model}")


def compute_component_centroids_xy_m(
    component_labels: Int64Array,
    cell_positions_xyz: Float64Array,
) -> Float64Array:
    """Average the position of every cell sharing a component label.

    Args:
        component_labels: Int64 array of shape (cell_count,), -1 off the front.
        cell_positions_xyz: Float64 array of shape (cell_count, 3).

    Returns:
        Float64 array of shape (n_components, 2), ordered by label.
    """
    is_labelled = component_labels >= 0
    if not np.any(is_labelled):
        return np.empty((0, 2), dtype=np.float64)

    labels = component_labels[is_labelled]
    positions_xy_m = cell_positions_xyz[is_labelled, :2]
    component_count = int(labels.max()) + 1
    cell_counts = np.bincount(labels, minlength=component_count)
    summed_x_m = np.bincount(
        labels, weights=positions_xy_m[:, 0], minlength=component_count
    )
    summed_y_m = np.bincount(
        labels, weights=positions_xy_m[:, 1], minlength=component_count
    )
    return np.stack((summed_x_m / cell_counts, summed_y_m / cell_counts), axis=1)


def compute_output_sample_count(
    mesh: SquareGridMesh,
    receiver_positions_xy_m: Float64Array,
    conditions: AtmosphericConditions,
    segment_duration_s: float,
    sample_rate_hz: int,
) -> int:
    """Fixed sample count every rendered timestep is padded or trimmed to.

    `render_receiver_signals` appends propagation delay to each channel, so
    blocks differ in length between sources and between timesteps. Sizing
    once against the worst-case source-receiver distance in the domain keeps
    the saved array rectangular without discarding any delayed arrival.

    Args:
        mesh: The grid, supplying the set of possible source positions.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        conditions: Air state, setting the speed of sound.
        segment_duration_s: Length of the undelayed signal, in seconds.
        sample_rate_hz: Samples per second.

    Returns:
        Sample count covering the segment plus the longest propagation delay.
    """
    cell_positions_xy_m = mesh.cell_positions_xyz[:, :2]
    maximum_distance_m = float(
        np.max(
            np.linalg.norm(
                cell_positions_xy_m[:, None, :] - receiver_positions_xy_m[None, :, :],
                axis=2,
            )
        )
    )
    speed_of_sound_m_per_s = compute_speed_of_sound_m_per_s(
        conditions.air_temperature_celsius
    )
    delay_sample_count = int(
        np.ceil(maximum_distance_m / speed_of_sound_m_per_s * sample_rate_hz)
    )
    return int(segment_duration_s * sample_rate_hz) + delay_sample_count + 2


def fit_to_sample_count(signals: Float64Array, sample_count: int) -> Float64Array:
    """Pad with zeros or trim so a block has exactly the requested length.

    Args:
        signals: Float64 array of shape (n_receivers, n_samples).
        sample_count: Target number of samples.

    Returns:
        Float64 array of shape (n_receivers, sample_count).
    """
    if signals.shape[1] >= sample_count:
        return signals[:, :sample_count]
    return np.pad(signals, ((0, 0), (0, sample_count - signals.shape[1])))


def render_one_timestep(
    sources: BurningCellSources,
    receiver_positions_xy_m: Float64Array,
    conditions: AtmosphericConditions,
    config: ForwardSimulationConfiguration,
    output_sample_count: int,
) -> Float64Array:
    """Sum every radiating cell's contribution at every receiver.

    Sources are summed one at a time because `render_receiver_signals` takes
    a single source position. That loop is over radiating cells, not over
    mesh cells, and it is the cost driver of the whole script.

    Args:
        sources: Positions, amplitudes and mesh indices of the radiating cells.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        conditions: Air state for absorption and speed of sound.
        config: Run configuration, for sampling and reference distance.
        output_sample_count: Length every returned block is fitted to.

    Returns:
        Float64 array of shape (n_receivers, output_sample_count).
    """
    combined_signals = np.zeros(
        (receiver_positions_xy_m.shape[0], output_sample_count), dtype=np.float64
    )
    for cell_local_index in range(sources.source_positions_xy_m.shape[0]):
        source_signal = generate_source_signal_for_burning_cell(
            float(sources.source_amplitudes[cell_local_index]),
            config.acoustic.segment_duration_s,
            config.acoustic.sample_rate_hz,
            seed=config.acoustic.source_signal_seed
            + int(sources.burning_cell_indices[cell_local_index]),
        )
        receiver_signals = render_receiver_signals(
            source_signal,
            config.acoustic.sample_rate_hz,
            sources.source_positions_xy_m[cell_local_index],
            receiver_positions_xy_m,
            conditions,
            config.acoustic.reference_distance_m,
        )
        combined_signals += fit_to_sample_count(receiver_signals, output_sample_count)
    return combined_signals


def build_ground_truth_record(
    fire_state: FireState,
    mesh: SquareGridMesh,
    sources: BurningCellSources,
) -> dict[str, Any]:
    """Describe where the fire actually is at one timestep.

    Components are labelled on the advancing front regardless of the emission
    model, so the ground truth answers "how many separate fires" identically
    for both.

    Args:
        fire_state: State to read burning and ignited cells from.
        mesh: The grid, supplying positions and connectivity.
        sources: The radiating cells this timestep rendered.

    Returns:
        Mapping ready to serialise into ground_truth.json.
    """
    is_on_front = compute_fire_front_mask(fire_state, mesh.neighbor_indices)
    component_labels = identify_connected_front_components(
        is_on_front, mesh.neighbor_indices
    )
    centroids_xy_m = compute_component_centroids_xy_m(
        component_labels, mesh.cell_positions_xyz
    )
    return {
        "simulation_time_s": float(fire_state.current_time_s),
        "component_count": int(centroids_xy_m.shape[0]),
        "component_centroids_xy_m": centroids_xy_m.tolist(),
        "front_cell_count": int(np.count_nonzero(is_on_front)),
        "burning_cell_count": int(np.count_nonzero(fire_state.is_burning)),
        "source_count": int(sources.burning_cell_indices.size),
    }


def build_mesh(config: ForwardSimulationConfiguration) -> SquareGridMesh:
    """Build the grid the fire runs on.

    Args:
        config: Run configuration, for extent, spacing and connectivity.

    Returns:
        The mesh.
    """
    return SquareGridMesh(
        SquareGridMeshConfig(
            extent_x_m=config.mesh.extent_x_m,
            extent_y_m=config.mesh.extent_y_m,
            cell_spacing_m=config.mesh.cell_spacing_m,
            use_diagonal_neighbors=config.mesh.use_diagonal_neighbors,
        )
    )


def write_run(
    output_directory: Path,
    config: ForwardSimulationConfiguration,
    receiver_positions_xy_m: Float64Array,
    receiver_signals: Float64Array,
    ground_truth: list[dict[str, Any]],
    time_step_s: float,
) -> None:
    """Save one run to disk.

    Args:
        output_directory: Directory to create and write into.
        config: The resolved run configuration.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        receiver_signals: Float64 array of shape (n_steps, n_receivers, n_samples).
        ground_truth: One record per timestep.
        time_step_s: The timestep the run used, in seconds.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    np.save(output_directory / "receiver_positions_xy_m.npy", receiver_positions_xy_m)
    np.save(output_directory / "receiver_signals.npy", receiver_signals)
    (output_directory / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "metadata.json").write_text(
        json.dumps(
            {
                "time_step_s": time_step_s,
                "time_step_count": int(receiver_signals.shape[0]),
                "sample_rate_hz": config.acoustic.sample_rate_hz,
                "receiver_count": int(receiver_positions_xy_m.shape[0]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> Path:
    """Run the simulation the configuration directory describes and save it.

    Returns:
        Directory the run was written to.
    """
    arguments = parse_arguments()
    config = load_forward_simulation_configuration(arguments.configs)

    mesh = build_mesh(config)
    fuel = build_fuel(config.fire.fuel_preset_name)
    wind_field = ConstantWindField.from_speed_and_bearing(
        config.wind.wind_speed_m_per_s, config.wind.wind_bearing_rad
    )
    fuel_field = UniformScalarField(fuel.fuel_load_kg_per_m2)

    maximum_rate_of_spread_m_per_s = float(
        compute_rate_of_spread_balbi_2009(
            np.array([config.wind.wind_speed_m_per_s]), np.zeros(1), fuel
        )[0]
    )
    time_step_s = compute_maximum_stable_time_step_s(
        mesh,
        maximum_rate_of_spread_m_per_s,
        config.fire.time_step_safety_factor,
        fuel.residence_time_s,
    )

    engine = RateOfSpreadEngine(fuel)
    fire_state = engine.initialize(mesh, fuel_field, wind_field)
    fire_state = engine.ignite_cells(
        fire_state,
        np.array(
            [
                compute_ignition_cell_index(
                    mesh,
                    config.fire.ignition_cell_x_fraction,
                    config.fire.ignition_cell_y_fraction,
                )
            ],
            dtype=np.int64,
        ),
    )

    receiver_positions_xy_m = place_receivers_from_configuration(
        config.receiver, config.mesh.extent_x_m, config.mesh.extent_y_m
    )
    output_sample_count = compute_output_sample_count(
        mesh,
        receiver_positions_xy_m,
        config.atmosphere,
        config.acoustic.segment_duration_s,
        config.acoustic.sample_rate_hz,
    )

    time_step_count = max(1, int(config.fire.simulation_duration_s / time_step_s))
    print(
        f"dt = {time_step_s:.2f} s over {time_step_count} steps, "
        f"{receiver_positions_xy_m.shape[0]} receivers, "
        f"{output_sample_count} samples per segment, "
        f"emission = {config.fire.emission_model}"
    )

    all_receiver_signals: list[Float64Array] = []
    ground_truth: list[dict[str, Any]] = []
    for step_index in range(time_step_count):
        fire_state = engine.step(fire_state, time_step_s)
        sources = extract_sources(config.fire.emission_model, fire_state, mesh, fuel)
        if sources.burning_cell_indices.size == 0:
            all_receiver_signals.append(
                np.zeros(
                    (receiver_positions_xy_m.shape[0], output_sample_count),
                    dtype=np.float64,
                )
            )
        else:
            all_receiver_signals.append(
                render_one_timestep(
                    sources,
                    receiver_positions_xy_m,
                    config.atmosphere,
                    config,
                    output_sample_count,
                )
            )
        record = build_ground_truth_record(fire_state, mesh, sources)
        ground_truth.append(record)
        if step_index % PROGRESS_REPORT_INTERVAL_STEPS == 0:
            print(
                f"step {step_index}/{time_step_count}, "
                f"t = {record['simulation_time_s']:.1f} s, "
                f"burning = {record['burning_cell_count']}, "
                f"front = {record['front_cell_count']}, "
                f"components = {record['component_count']}"
            )

    output_directory = OUTPUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    write_run(
        output_directory,
        config,
        receiver_positions_xy_m,
        np.stack(all_receiver_signals),
        ground_truth,
        time_step_s,
    )
    print(output_directory)
    return output_directory


if __name__ == "__main__":
    main()
