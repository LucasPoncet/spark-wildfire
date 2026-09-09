"""Runs a fire simulation and renders receiver signals at fixed observations.

Usage:
    uv run python scripts/run_acoustic_rendering.py
    uv run python scripts/run_acoustic_rendering.py --configs other_dir

This is the handoff to the inverse team. The fire steps at its own
physics-bounded timestep; the array listens every `observation_interval_s`
of simulated time, which is an experimental choice and not a physical one.
Serialisation belongs to `simulation_run_writer`, so nothing here formats a
file. Zero domain logic lives here either.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config.acoustic_rendering_configuration import compute_observation_stride
from src.config.fire_simulation_configuration import (
    ALL_BURNING_EMISSION,
    FRONT_ONLY_EMISSION,
)
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    ForwardSimulationConfiguration,
    load_forward_simulation_configuration,
)
from src.config.simulation_context_factory import build_simulation_context
from src.spark.acoustic.burning_cell_source_model import (
    BurningCellSources,
    compute_fire_front_mask,
    extract_burning_cell_sources,
    extract_fire_front_sources,
    generate_source_signal_for_burning_cell,
    identify_connected_front_components,
)
from src.spark.acoustic.free_field_propagation import render_receiver_signals
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.spark.fire.fire_state import FireState
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.terrain.mesh_protocol import MeshProtocol
from src.utils.array_types import Float32Array, Float64Array, Int64Array
from src.utils.io.simulation_run_writer import write_simulation_run

OUTPUT_ROOT: Path = Path("results/simulations")
PROGRESS_REPORT_INTERVAL_OBSERVATIONS: int = 5
MINIMUM_RECEIVER_RANGE_SPREAD_M: float = 1.0


def parse_arguments() -> argparse.Namespace:
    """Parses the only argument this script takes.

    Returns:
        The parsed arguments, carrying the configuration directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    return parser.parse_args()


def extract_sources(
    emission_model: str,
    fire_state: FireState,
    mesh: MeshProtocol,
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
    mesh: MeshProtocol,
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


def render_one_observation(
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
    mesh: MeshProtocol,
    sources: BurningCellSources,
    receiver_positions_xy_m: Float64Array,
    ignition_positions_xy_m: Float64Array,
    cell_area_m2: float,
) -> dict[str, Any]:
    """Describe where the fire actually is at one observation.

    Components are labelled on the advancing front regardless of the emission
    model, so the ground truth answers "how many separate fires" identically
    for both.

    Args:
        fire_state: State to read burning and ignited cells from.
        mesh: The grid, supplying positions and connectivity.
        sources: The radiating cells this observation rendered.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        ignition_positions_xy_m: Float64 array of shape (k, 2), where the
            fires were lit, used to report how far the front has travelled.
        cell_area_m2: Ground area one cell covers, in square metres.

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
    front_positions_xy_m = mesh.cell_positions_xyz[is_on_front, :2]
    front_centroid_xy_m = (
        front_positions_xy_m.mean(axis=0)
        if front_positions_xy_m.shape[0] > 0
        else np.full(2, np.nan)
    )
    return {
        "simulation_time_s": float(fire_state.current_time_s),
        "component_count": int(centroids_xy_m.shape[0]),
        "component_centroids_xy_m": centroids_xy_m.tolist(),
        "front_centroid_xy_m": front_centroid_xy_m.tolist(),
        "front_radius_m": compute_front_radius_m(
            front_positions_xy_m, ignition_positions_xy_m
        ),
        "front_cell_count": int(np.count_nonzero(is_on_front)),
        "burning_cell_count": int(np.count_nonzero(fire_state.is_burning)),
        "ignited_cell_count": int(np.count_nonzero(fire_state.has_ignited)),
        "burnt_area_m2": float(np.count_nonzero(fire_state.has_ignited) * cell_area_m2),
        "source_count": int(sources.burning_cell_indices.size),
        "receiver_range_spread_m": compute_receiver_range_spread_m(
            front_centroid_xy_m.reshape(1, 2)
            if front_positions_xy_m.shape[0] > 0
            else np.empty((0, 2)),
            receiver_positions_xy_m,
        ),
    }


def compute_front_radius_m(
    front_positions_xy_m: Float64Array,
    ignition_positions_xy_m: Float64Array,
) -> float:
    """Mean distance from each front cell to its nearest ignition point.

    Grows monotonically while the fire spreads, unlike any one component's
    centroid, which jumps between fragments when the burning band is thinner
    than the gap between two burning cells.

    Args:
        front_positions_xy_m: Float64 array of shape (n_front, 2).
        ignition_positions_xy_m: Float64 array of shape (k, 2).

    Returns:
        Mean travelled distance in metres, or 0.0 with no front.
    """
    if front_positions_xy_m.shape[0] == 0:
        return 0.0
    distances_m = np.linalg.norm(
        front_positions_xy_m[:, None, :] - ignition_positions_xy_m[None, :, :], axis=2
    )
    return float(distances_m.min(axis=1).mean())


def compute_receiver_range_spread_m(
    centroids_xy_m: Float64Array,
    receiver_positions_xy_m: Float64Array,
) -> float:
    """Spread of source-receiver ranges from the largest front component.

    Zero means every receiver sits at the same distance from the fire, so
    every level ratio is one and every time difference of arrival is zero.
    Such a scene carries no information for any estimator.

    Args:
        centroids_xy_m: Float64 array of shape (n_components, 2), largest first.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).

    Returns:
        Largest minus smallest range, in metres, or 0.0 with no front.
    """
    if centroids_xy_m.shape[0] == 0:
        return 0.0
    ranges_m = np.linalg.norm(receiver_positions_xy_m - centroids_xy_m[0], axis=1)
    return float(ranges_m.max() - ranges_m.min())


def verify_geometry_is_informative(
    ignition_xy_m: Float64Array,
    receiver_positions_xy_m: Float64Array,
    minimum_range_spread_m: float,
) -> None:
    """Refuse to render a scene whose receivers are equidistant from the fire.

    Checked against the ignition point before any rendering starts, because a
    front spreading symmetrically from the centre of a receiver ring stays
    equidistant for the whole run.

    Args:
        ignition_xy_m: Float64 array of shape (2,), where the fire starts.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        minimum_range_spread_m: Smallest accepted spread of ranges, in metres.

    Raises:
        ValueError: If the ranges are too close to equal to carry information.
    """
    range_spread_m = compute_receiver_range_spread_m(
        ignition_xy_m.reshape(-1, 2), receiver_positions_xy_m
    )
    if range_spread_m < minimum_range_spread_m:
        raise ValueError(
            f"receiver ranges span only {range_spread_m:.3f} m around the ignition "
            f"point, below the {minimum_range_spread_m} m minimum; this geometry is "
            f"degenerate and carries no information"
        )


def main() -> Path:
    """Run the simulation the configuration directory describes and save it.

    Returns:
        Directory the run was written to.
    """
    arguments = parse_arguments()
    config = load_forward_simulation_configuration(arguments.configs)
    context = build_simulation_context(config)

    maximum_rate_of_spread_m_per_s = float(
        compute_rate_of_spread_balbi_2009(
            np.array([config.wind.wind_speed_m_per_s]), np.zeros(1), context.fuel
        )[0]
    )
    time_step_s = compute_maximum_stable_time_step_s(
        context.mesh,
        maximum_rate_of_spread_m_per_s,
        config.fire.time_step_safety_factor,
        context.fuel.residence_time_s,
    )
    observation_stride = compute_observation_stride(
        config.acoustic.observation_interval_s, time_step_s
    )

    fire_state = context.spread_engine.initialize(
        context.mesh, context.fuel_field, context.wind_field
    )
    fire_state = context.spread_engine.ignite_cells(
        fire_state, context.ignition_cell_indices
    )

    ignition_positions_xy_m = context.mesh.cell_positions_xyz[
        context.ignition_cell_indices, :2
    ]
    verify_geometry_is_informative(
        ignition_positions_xy_m,
        context.receiver_positions_xy_m,
        MINIMUM_RECEIVER_RANGE_SPREAD_M,
    )
    output_sample_count = compute_output_sample_count(
        context.mesh,
        context.receiver_positions_xy_m,
        config.atmosphere,
        config.acoustic.segment_duration_s,
        config.acoustic.sample_rate_hz,
    )

    observation_count = max(
        1,
        int(config.fire.simulation_duration_s / config.acoustic.observation_interval_s),
    )
    print(
        f"{config.experiment.name}: {config.experiment.title}\n"
        f"{config.fire.spread_engine_name} engine, dt = {time_step_s:.2f} s, "
        f"observing every {observation_stride} steps "
        f"({config.acoustic.observation_interval_s:.0f} s) for {observation_count} "
        f"observations, {context.receiver_positions_xy_m.shape[0]} receivers, "
        f"{output_sample_count} samples per block, "
        f"emission = {config.fire.emission_model}"
    )

    all_receiver_signals: list[Float64Array] = []
    ground_truth: list[dict[str, Any]] = []
    for observation_index in range(observation_count):
        for _ in range(observation_stride):
            fire_state = context.spread_engine.step(fire_state, time_step_s)
        sources = extract_sources(
            config.fire.emission_model, fire_state, context.mesh, context.fuel
        )
        if sources.burning_cell_indices.size == 0:
            all_receiver_signals.append(
                np.zeros(
                    (context.receiver_positions_xy_m.shape[0], output_sample_count),
                    dtype=np.float64,
                )
            )
        else:
            all_receiver_signals.append(
                render_one_observation(
                    sources,
                    context.receiver_positions_xy_m,
                    config.atmosphere,
                    config,
                    output_sample_count,
                )
            )
        record = build_ground_truth_record(
            fire_state,
            context.mesh,
            sources,
            context.receiver_positions_xy_m,
            ignition_positions_xy_m,
            config.mesh.cell_spacing_m**2,
        )
        ground_truth.append(record)
        if observation_index % PROGRESS_REPORT_INTERVAL_OBSERVATIONS == 0:
            print(
                f"observation {observation_index}/{observation_count}, "
                f"t = {record['simulation_time_s']:.0f} s, "
                f"burning = {record['burning_cell_count']}, "
                f"front = {record['front_cell_count']}, "
                f"components = {record['component_count']}, "
                f"radius = {record['front_radius_m']:.1f} m, "
                f"range spread = {record['receiver_range_spread_m']:.1f} m"
            )

    receiver_signals: Float32Array = np.stack(all_receiver_signals).astype(np.float32)
    output_directory = (
        OUTPUT_ROOT
        / config.experiment.name
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    write_simulation_run(
        output_directory,
        config.to_dict(),
        context.receiver_positions_xy_m,
        receiver_signals,
        ground_truth,
        {
            "experiment_name": config.experiment.name,
            "experiment_title": config.experiment.title,
            "configuration_directory": str(arguments.configs),
            "time_step_s": time_step_s,
            "observation_interval_s": config.acoustic.observation_interval_s,
            "observation_count": int(receiver_signals.shape[0]),
            "sample_rate_hz": config.acoustic.sample_rate_hz,
            "receiver_count": int(context.receiver_positions_xy_m.shape[0]),
            "spread_engine_name": config.fire.spread_engine_name,
            "emission_model": config.fire.emission_model,
            "ignition_positions_xy_m": ignition_positions_xy_m.tolist(),
        },
    )
    print(output_directory)
    return output_directory


if __name__ == "__main__":
    main()
