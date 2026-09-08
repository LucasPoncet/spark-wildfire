"""Runs a fire simulation and renders receiver signals at every timestep.

Usage:
    uv run python scripts/run_acoustic_rendering.py [configs/default.json]

This is the handoff to the inverse team: it writes receiver positions, a
`(timestep, receiver, sample)` signal array, the resolved configuration and
a metadata sidecar into `results/simulation_runs/<timestamp>/`.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config.fire_rendering_configuration import FireRenderingConfiguration
from src.spark.acoustic.burning_cell_source_model import (
    extract_burning_cell_sources,
    generate_source_signal_for_burning_cell,
)
from src.spark.acoustic.free_field_propagation import render_receiver_signals
from src.spark.acoustic.receiver_placement import place_receivers_from_configuration
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_engine import RateOfSpreadEngine
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.terrain.square_grid_mesh import SquareGridMesh, SquareGridMeshConfig
from src.utils.array_types import Float64Array

DEFAULT_CONFIGURATION_PATH = Path("configs/default.json")
OUTPUT_ROOT = Path("results/simulation_runs")
PROGRESS_REPORT_INTERVAL_STEPS = 50
FUEL_PRESETS = {"pine_needle_litter": FuelProperties.pine_needle_litter}


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
    sources_positions_xy_m: Float64Array,
    source_amplitudes: Float64Array,
    burning_cell_indices: Float64Array,
    receiver_positions_xy_m: Float64Array,
    conditions: AtmosphericConditions,
    config: FireRenderingConfiguration,
    output_sample_count: int,
) -> Float64Array:
    """Sum every burning cell's contribution at every receiver.

    Sources are summed one at a time because `render_receiver_signals` takes
    a single source position. That loop is over burning cells, not over mesh
    cells, and it is the cost driver of the whole script.

    Args:
        sources_positions_xy_m: Float64 array of shape (n_burning, 2).
        source_amplitudes: Float64 array of shape (n_burning,).
        burning_cell_indices: Mesh indices of the burning cells, used to give
            each cell its own reproducible noise seed.
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
    for cell_local_index in range(sources_positions_xy_m.shape[0]):
        source_signal = generate_source_signal_for_burning_cell(
            float(source_amplitudes[cell_local_index]),
            config.acoustic.segment_duration_s,
            config.acoustic.sample_rate_hz,
            seed=config.acoustic.source_signal_seed
            + int(burning_cell_indices[cell_local_index]),
        )
        receiver_signals = render_receiver_signals(
            source_signal,
            config.acoustic.sample_rate_hz,
            sources_positions_xy_m[cell_local_index],
            receiver_positions_xy_m,
            conditions,
            config.acoustic.reference_distance_m,
        )
        combined_signals += fit_to_sample_count(receiver_signals, output_sample_count)
    return combined_signals


def main(config_path: Path) -> Path:
    """Run the simulation described by a configuration file and save the render.

    Args:
        config_path: JSON configuration to read.

    Returns:
        Directory the run was written to.
    """
    config = FireRenderingConfiguration.from_json(config_path)

    mesh = SquareGridMesh(
        SquareGridMeshConfig(
            extent_x_m=config.mesh.extent_x_m,
            extent_y_m=config.mesh.extent_y_m,
            cell_spacing_m=config.mesh.cell_spacing_m,
            use_diagonal_neighbors=config.mesh.use_diagonal_neighbors,
        )
    )
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
    conditions = AtmosphericConditions(
        air_temperature_celsius=config.atmosphere.air_temperature_celsius,
        relative_humidity_percent=config.atmosphere.relative_humidity_percent,
        pressure_kpa=config.atmosphere.atmospheric_pressure_kpa,
    )
    output_sample_count = compute_output_sample_count(
        mesh,
        receiver_positions_xy_m,
        conditions,
        config.acoustic.segment_duration_s,
        config.acoustic.sample_rate_hz,
    )

    time_step_count = max(1, int(config.fire.simulation_duration_s / time_step_s))
    print(
        f"dt = {time_step_s:.2f} s over {time_step_count} steps, "
        f"{receiver_positions_xy_m.shape[0]} receivers, "
        f"{output_sample_count} samples per segment"
    )

    all_receiver_signals: list[Float64Array] = []
    for step_index in range(time_step_count):
        fire_state = engine.step(fire_state, time_step_s)
        sources = extract_burning_cell_sources(
            fire_state,
            mesh.cell_positions_xyz,
            fuel.fuel_load_kg_per_m2,
            fuel.residence_time_s,
        )
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
                    sources.source_positions_xy_m,
                    sources.source_amplitudes,
                    sources.burning_cell_indices,
                    receiver_positions_xy_m,
                    conditions,
                    config,
                    output_sample_count,
                )
            )
        if step_index % PROGRESS_REPORT_INTERVAL_STEPS == 0:
            print(
                f"step {step_index}/{time_step_count}, "
                f"t = {step_index * time_step_s:.1f} s, "
                f"burning = {sources.burning_cell_indices.size}"
            )

    output_directory = OUTPUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_directory.mkdir(parents=True, exist_ok=True)
    config.to_json(output_directory / "config.json")
    np.save(output_directory / "receiver_positions_xy_m.npy", receiver_positions_xy_m)
    np.save(output_directory / "receiver_signals.npy", np.stack(all_receiver_signals))
    (output_directory / "metadata.json").write_text(
        json.dumps(
            {
                "time_step_s": time_step_s,
                "time_step_count": time_step_count,
                "sample_rate_hz": config.acoustic.sample_rate_hz,
                "receiver_count": int(receiver_positions_xy_m.shape[0]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_directory)
    return output_directory


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIGURATION_PATH)
