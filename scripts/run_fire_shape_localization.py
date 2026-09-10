"""Estimates the shape of a spreading fire from twenty receivers, window by window.

Usage:
    uv run python scripts/run_fire_shape_localization.py --configs configs/f1
    uv run python scripts/run_fire_shape_localization.py --configs configs/f1 \
        --max-observations 2

Joins the two halves of the repository that have so far only met through a
saved run directory. The fire spreads under the propagation-equation engine;
every burning cell on the front radiates its own broadband waveform; the ring
hears the mixture; and the multi-source estimator is handed those waveforms
with no idea how many cells produced them.

The source count is never the deliverable here and cannot be: a front carries
tens of cells and no sequential deflation loop recovers that many. What the
frames show is the steered response power map, whose ridge is the shape
estimate, with the handful of positions the loop accepted marked on it.

Zero domain logic lives here. The emission-model dispatch is imported from
`run_acoustic_rendering` rather than restated, so the two runners cannot drift
about which cells radiate.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from scripts.run_acoustic_rendering import extract_sources
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    ForwardSimulationConfiguration,
    SimulationConfiguration,
    load_forward_simulation_configuration,
    load_simulation_configuration,
)
from src.config.simulation_context import SimulationContext
from src.config.simulation_context_factory import build_simulation_context
from src.spark.acoustic.burning_cell_source_model import (
    BurningCellSources,
    compute_fire_front_mask,
    generate_source_signal_for_burning_cell,
    identify_connected_front_components,
)
from src.spark.acoustic.free_field_propagation import (
    render_multi_source_receiver_signals,
)
from src.spark.acoustic.receiver_noise import add_white_noise_to_receiver_signals
from src.spark.acoustic.receiver_placement import place_receivers_from_layout
from src.spark.fire.cellular_automaton_spread_engine import (
    FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
)
from src.spark.fire.fire_state import FireState
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.spark.fire.spread_engine_protocol import SpreadEngineProtocol
from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.inverse.multiple_source_estimator import (
    MultiSourceLocalization,
    localize_multiple_sources,
)
from src.spark.terrain.square_grid_mesh import SquareGridMesh
from src.utils.array_types import Float64Array
from src.utils.io.metrics_writer import write_metrics_document
from src.utils.visualization.fire_shape_plotter import (
    plot_fire_and_steered_response_power,
)

FIGURE_ROOT: Path = Path("results/figures")
FRONT_COVERAGE_RADIUS_M: float = 4.0
ANIMATION_FRAME_DURATION_MS: int = 900


@dataclass(frozen=True)
class ObservationResult:
    """One 5 s window: what burned, what was heard, and what was found.

    Attributes:
        simulation_time_s: Simulated time at the end of the window.
        fire_state: Fire state the window was rendered from.
        radiating_positions_xy_m: Cells the emission model sounded, shape
            `(n_radiating, 2)`. Under `front_only` these are the front; under
            `all_burning` they are the whole burning area, which is why the
            count is kept apart from `front_cell_count` rather than assumed
            equal to it.
        front_cell_count: Cells on the advancing perimeter, from the front mask
            and independent of which model radiates.
        front_component_count: Separate fragments that mask reports.
        burning_cell_count: Cells alight, radiating or not.
        localization: What the estimator returned.
    """

    simulation_time_s: float
    fire_state: FireState
    radiating_positions_xy_m: Float64Array
    front_cell_count: int
    front_component_count: int
    burning_cell_count: int
    localization: MultiSourceLocalization


def parse_arguments() -> argparse.Namespace:
    """Reads the configuration directory and the optional observation cap.

    Returns:
        Parsed arguments carrying `configs` and `max_observations`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    parser.add_argument(
        "--max-observations",
        type=int,
        default=None,
        help="Stop after this many windows. Smoke-tests the whole path in one "
        "window instead of the ten the scene declares.",
    )
    return parser.parse_args()


def resolve_time_step_s(
    config: ForwardSimulationConfiguration, context: SimulationContext
) -> float:
    """Derives the fire timestep from the CFL condition and the residence time.

    Args:
        config: Run configuration, supplying wind and the safety factor.
        context: The built context, supplying mesh and fuel.

    Returns:
        Timestep in seconds.
    """
    maximum_rate_of_spread_m_per_s = float(
        compute_rate_of_spread_balbi_2009(
            np.array([config.wind.wind_speed_m_per_s]), np.zeros(1), context.fuel
        )[0]
    )
    return compute_maximum_stable_time_step_s(
        context.mesh,
        maximum_rate_of_spread_m_per_s,
        config.fire.time_step_safety_factor,
        context.fuel.residence_time_s,
    )


def advance_fire_to_time_s(
    spread_engine: SpreadEngineProtocol,
    fire_state: FireState,
    target_time_s: float,
    time_step_s: float,
) -> FireState:
    """Steps the engine until the clock reaches the end of this window.

    The last step is shortened rather than overshot, so the rendered instant is
    exactly the observation time the frames are labelled with. Overshooting
    would be silent: the frame would still be captioned with the time it was
    asked for while showing the fire some fraction of a step later.

    Args:
        spread_engine: The propagation model.
        fire_state: State to advance.
        target_time_s: Simulated time to stop at.
        time_step_s: Largest step the CFL condition permits.

    Returns:
        The advanced state.
    """
    while fire_state.current_time_s < target_time_s - 1e-9:
        remaining_s = target_time_s - fire_state.current_time_s
        fire_state = spread_engine.step(fire_state, min(time_step_s, remaining_s))
    return fire_state


def lift_sources_to_height_xyz_m(
    sources: BurningCellSources, source_height_m: float
) -> Float64Array:
    """Puts the radiating cells at the configured flame height.

    Args:
        sources: The radiating cells.
        source_height_m: Height above the ground plane, in metres.

    Returns:
        Positions of shape `(n_sources, 3)`.
    """
    positions_xy_m = np.atleast_2d(
        np.asarray(sources.source_positions_xy_m, dtype=np.float64)
    )
    return np.column_stack(
        (
            positions_xy_m,
            np.full(positions_xy_m.shape[0], source_height_m, dtype=np.float64),
        )
    )


def render_front_to_receivers(
    sources: BurningCellSources,
    receiver_positions_xyz_m: Float64Array,
    source_height_m: float,
    config: ForwardSimulationConfiguration,
    signal_to_noise_ratio_db: float | None,
    observation_index: int,
) -> Float64Array:
    """Sounds every radiating cell at once and adds sensor noise.

    Each cell is given its own seeded broadband waveform, keyed on the cell's
    mesh index so that a cell sounds the same whenever it is alight and no two
    cells share content. Amplitude is carried by the scale rather than baked
    into the waveform, so the source model stays the one place that decides how
    loud a burning cell is.

    The noise generator is reseeded per window rather than per run, so a window
    is reproducible on its own and two windows do not share a noise realisation.

    Args:
        sources: The radiating cells.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        source_height_m: Height of every source above the ground plane.
        config: Run configuration, supplying sampling and the reference distance.
        signal_to_noise_ratio_db: Requested ratio, or None when noise is off.
        observation_index: Which window this is, seeding the noise.

    Returns:
        Receiver channels of shape `(n_receivers, n_samples)`.
    """
    sample_count = round(
        config.acoustic.segment_duration_s * config.acoustic.sample_rate_hz
    )
    if sources.burning_cell_indices.size == 0:
        return np.zeros(
            (receiver_positions_xyz_m.shape[0], sample_count), dtype=np.float64
        )
    source_signals = [
        generate_source_signal_for_burning_cell(
            1.0,
            config.acoustic.segment_duration_s,
            config.acoustic.sample_rate_hz,
            seed=config.acoustic.source_signal_seed + int(cell_index),
        )
        for cell_index in sources.burning_cell_indices
    ]
    receiver_signals = render_multi_source_receiver_signals(
        source_signals,
        lift_sources_to_height_xyz_m(sources, source_height_m),
        np.asarray(sources.source_amplitudes, dtype=np.float64),
        receiver_positions_xyz_m,
        config.acoustic.sample_rate_hz,
        config.atmosphere,
        config.acoustic.reference_distance_m,
    )
    if signal_to_noise_ratio_db is None:
        return receiver_signals
    return add_white_noise_to_receiver_signals(
        receiver_signals,
        signal_to_noise_ratio_db,
        np.random.default_rng(config.acoustic.source_signal_seed + observation_index),
    )


def compute_distances_to_front_m(
    estimated_positions_xy_m: Float64Array, front_positions_xy_m: Float64Array
) -> Float64Array:
    """Distance from each estimate to the nearest burning cell.

    With an unknown number of sources spread along a contour there is no pairing
    to score, so an estimate is judged by whether it landed on the fire at all.

    Args:
        estimated_positions_xy_m: Accepted sources, shape `(n_sources, 2)`.
        front_positions_xy_m: Radiating cells, shape `(n_front_cells, 2)`.

    Returns:
        Distances in metres, shape `(n_sources,)`.
    """
    estimates = np.atleast_2d(np.asarray(estimated_positions_xy_m, dtype=np.float64))
    front = np.atleast_2d(np.asarray(front_positions_xy_m, dtype=np.float64))
    if estimates.size == 0 or front.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.min(
        np.linalg.norm(estimates[:, None, :] - front[None, :, :], axis=2), axis=1
    )


def compute_front_coverage_fraction(
    estimated_positions_xy_m: Float64Array,
    front_positions_xy_m: Float64Array,
    coverage_radius_m: float,
) -> float:
    """Share of the burning contour that has an estimate near it.

    The complement of the distance measure: one estimate sitting exactly on a
    twelve-metre front is accurate and says almost nothing about its shape.

    Args:
        estimated_positions_xy_m: Accepted sources, shape `(n_sources, 2)`.
        front_positions_xy_m: Radiating cells, shape `(n_front_cells, 2)`.
        coverage_radius_m: How near an estimate must be to cover a cell.

    Returns:
        Fraction of front cells within the radius of some estimate.
    """
    estimates = np.atleast_2d(np.asarray(estimated_positions_xy_m, dtype=np.float64))
    front = np.atleast_2d(np.asarray(front_positions_xy_m, dtype=np.float64))
    if estimates.size == 0 or front.size == 0:
        return 0.0
    nearest_m = np.min(
        np.linalg.norm(front[:, None, :] - estimates[None, :, :], axis=2), axis=1
    )
    return float(np.mean(nearest_m <= coverage_radius_m))


def stack_estimated_positions_xy_m(
    localization: MultiSourceLocalization,
) -> Float64Array:
    """Collects the accepted positions into one array.

    Args:
        localization: What the estimator returned.

    Returns:
        Positions of shape `(n_sources, 2)`, empty when nothing was accepted.
    """
    if not localization.sources:
        return np.empty((0, 2), dtype=np.float64)
    return np.stack(
        [
            np.asarray(source.position.position_xy_m, dtype=np.float64)
            for source in localization.sources
        ]
    )


def build_observation_record(
    result: ObservationResult, coverage_radius_m: float
) -> dict[str, Any]:
    """Reduces one window to the numbers the metrics document keeps.

    Args:
        result: Everything the window produced.
        coverage_radius_m: How near an estimate must be to cover a front cell.

    Returns:
        A JSON-serialisable mapping.
    """
    estimated_positions_xy_m = stack_estimated_positions_xy_m(result.localization)
    distances_m = compute_distances_to_front_m(
        estimated_positions_xy_m, result.radiating_positions_xy_m
    )
    return {
        "simulation_time_s": result.simulation_time_s,
        "burning_cell_count": result.burning_cell_count,
        "radiating_cell_count": int(result.radiating_positions_xy_m.shape[0]),
        "front_cell_count": result.front_cell_count,
        "front_component_count": result.front_component_count,
        "estimated_source_count": result.localization.estimated_source_count,
        "stop_reason": result.localization.stop_reason,
        "estimated_positions_xy_m": estimated_positions_xy_m.tolist(),
        "median_distance_to_front_m": (
            float(np.median(distances_m)) if distances_m.size else None
        ),
        "maximum_distance_to_front_m": (
            float(np.max(distances_m)) if distances_m.size else None
        ),
        "front_coverage_fraction": compute_front_coverage_fraction(
            estimated_positions_xy_m, result.radiating_positions_xy_m, coverage_radius_m
        ),
        "front_coverage_radius_m": coverage_radius_m,
    }


def build_frame(
    result: ObservationResult,
    mesh: SquareGridMesh,
    fuel_density_fraction: Float64Array,
    burn_duration_s: float,
    receiver_positions_xyz_m: Float64Array,
    experiment_title: str,
) -> Figure:
    """Draws one window as the two-panel frame.

    Args:
        result: Everything the window produced.
        mesh: The square grid the fire lives on.
        fuel_density_fraction: Fuel field sampled at every cell.
        burn_duration_s: How long a cell burns, used only to shade it.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        experiment_title: Scene title, used in the caption.

    Returns:
        The figure. This function does not save it.
    """
    return plot_fire_and_steered_response_power(
        result.fire_state,
        mesh,
        fuel_density_fraction,
        burn_duration_s,
        FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
        result.localization.first_round_map,
        result.localization.first_round_positions_xyz_m,
        receiver_positions_xyz_m,
        result.radiating_positions_xy_m,
        stack_estimated_positions_xy_m(result.localization),
        [
            np.asarray(source.position.position_covariance_m2, dtype=np.float64)
            for source in result.localization.sources
        ],
        f"{experiment_title} — t = {result.simulation_time_s:.0f} s, "
        f"{result.radiating_positions_xy_m.shape[0]} cells radiating, "
        f"{result.localization.estimated_source_count} located",
    )


def save_frame(figure: Figure, figure_directory: Path, observation_index: int) -> Path:
    """Writes one frame as a raster for the animation and a vector for the report.

    Args:
        figure: The frame.
        figure_directory: Directory to write into.
        observation_index: Which window this is.

    Returns:
        Path of the raster frame.
    """
    figure_directory.mkdir(parents=True, exist_ok=True)
    stem = f"fire_shape_frame_{observation_index:02d}"
    figure.savefig(figure_directory / f"{stem}.svg", bbox_inches="tight")
    raster_path = figure_directory / f"{stem}.png"
    figure.savefig(raster_path, dpi=140, bbox_inches="tight")
    return raster_path


def write_animation(frame_paths: list[Path], figure_directory: Path) -> Path:
    """Assembles the saved frames into one looping animation.

    Args:
        frame_paths: The raster frames, in observation order.
        figure_directory: Directory to write into.

    Returns:
        Path of the animation.

    Raises:
        ValueError: If there is no frame to animate.
    """
    if not frame_paths:
        raise ValueError("an animation needs at least one frame")
    frames = [
        Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE)
        for path in frame_paths
    ]
    path = figure_directory / "fire_shape_evolution.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=ANIMATION_FRAME_DURATION_MS,
        loop=0,
    )
    return path


def run_one_observation(
    observation_index: int,
    fire_state: FireState,
    context: SimulationContext,
    forward_config: ForwardSimulationConfiguration,
    configuration: SimulationConfiguration,
    receiver_positions_xyz_m: Float64Array,
    time_step_s: float,
) -> ObservationResult:
    """Advances the fire one window, sounds it, and localizes what comes back.

    Args:
        observation_index: Which window this is, counting from zero.
        fire_state: State at the start of the window.
        context: The built context.
        forward_config: Forward-side configuration.
        configuration: Localization-side configuration.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        time_step_s: Largest fire step the CFL condition permits.

    Returns:
        Everything the window produced.
    """
    target_time_s = (
        observation_index + 1
    ) * forward_config.acoustic.observation_interval_s
    fire_state = advance_fire_to_time_s(
        context.spread_engine, fire_state, target_time_s, time_step_s
    )
    sources = extract_sources(
        forward_config.fire.emission_model, fire_state, context.mesh, context.fuel
    )
    receiver_signals = render_front_to_receivers(
        sources,
        receiver_positions_xyz_m,
        configuration.geometry.source_height_m,
        forward_config,
        configuration.forward_model.receiver_noise.requested_signal_to_noise_ratio_db,
        observation_index,
    )
    localization = localize_multiple_sources(
        receiver_signals,
        receiver_positions_xyz_m,
        forward_config.acoustic.sample_rate_hz,
        configuration.geometry.domain.size_x_m,
        configuration.geometry.domain.size_y_m,
        forward_config.atmosphere,
        configuration.localization.multi_source,
    )
    front_mask = compute_fire_front_mask(fire_state, context.mesh.neighbor_indices)
    component_labels = identify_connected_front_components(
        front_mask, context.mesh.neighbor_indices
    )
    return ObservationResult(
        simulation_time_s=fire_state.current_time_s,
        fire_state=fire_state,
        radiating_positions_xy_m=np.asarray(
            sources.source_positions_xy_m, dtype=np.float64
        ),
        front_cell_count=int(np.count_nonzero(front_mask)),
        front_component_count=int(component_labels.max() + 1)
        if component_labels.size and component_labels.max() >= 0
        else 0,
        burning_cell_count=int(np.count_nonzero(fire_state.is_burning)),
        localization=localization,
    )


def print_observation_row(record: dict[str, Any]) -> None:
    """Prints one line of the run table.

    Args:
        record: What `build_observation_record` produced.
    """
    median_m = record["median_distance_to_front_m"]
    median_text = "     -" if median_m is None else f"{median_m:6.2f}"
    print(
        f"  t = {record['simulation_time_s']:5.1f} s  "
        f"burning {record['burning_cell_count']:4d}  "
        f"front {record['front_cell_count']:4d}  "
        f"radiating {record['radiating_cell_count']:4d}  "
        f"components {record['front_component_count']:2d}  "
        f"located {record['estimated_source_count']:2d}  "
        f"median off-front {median_text} m  "
        f"coverage {record['front_coverage_fraction']:5.1%}  "
        f"{record['stop_reason']}"
    )


def main() -> None:
    """Runs every window of the scene, saves the frames, writes the metrics."""
    arguments = parse_arguments()
    forward_config = load_forward_simulation_configuration(arguments.configs)
    configuration = load_simulation_configuration(arguments.configs)
    context = build_simulation_context(forward_config)

    mesh = context.mesh
    if not isinstance(mesh, SquareGridMesh):
        raise TypeError(
            "the fire panel renders a raster image and needs a square grid; "
            f"{type(mesh).__name__} has no row and column structure"
        )
    time_step_s = resolve_time_step_s(forward_config, context)
    receiver_positions_xyz_m = place_receivers_from_layout(
        configuration.geometry.receiver_layout,
        configuration.geometry.domain.size_x_m,
        configuration.geometry.domain.size_y_m,
    )
    fuel_density_fraction = np.clip(
        context.fuel_field.sample(mesh.cell_positions_xyz), 0.0, 1.0
    )

    fire_state = context.spread_engine.initialize(
        mesh, context.fuel_field, context.wind_field
    )
    fire_state = context.spread_engine.ignite_cells(
        fire_state, context.ignition_cell_indices
    )

    observation_count = max(
        1,
        int(
            forward_config.fire.simulation_duration_s
            / forward_config.acoustic.observation_interval_s
        ),
    )
    if arguments.max_observations is not None:
        observation_count = min(observation_count, arguments.max_observations)

    print(
        f"{forward_config.experiment.name}: {forward_config.experiment.title}\n"
        f"{forward_config.fire.spread_engine_name} engine, dt = {time_step_s:.2f} s, "
        f"{receiver_positions_xyz_m.shape[0]} receivers, "
        f"{observation_count} windows of "
        f"{forward_config.acoustic.observation_interval_s:.0f} s, "
        f"emission = {forward_config.fire.emission_model}, "
        f"deflation = {configuration.localization.multi_source.deflation.method}"
    )

    figure_directory = FIGURE_ROOT / forward_config.experiment.name
    records: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    for observation_index in range(observation_count):
        result = run_one_observation(
            observation_index,
            fire_state,
            context,
            forward_config,
            configuration,
            receiver_positions_xyz_m,
            time_step_s,
        )
        fire_state = result.fire_state
        record = build_observation_record(result, FRONT_COVERAGE_RADIUS_M)
        records.append(record)
        print_observation_row(record)
        frame_paths.append(
            save_frame(
                build_frame(
                    result,
                    mesh,
                    fuel_density_fraction,
                    context.fuel.residence_time_s,
                    receiver_positions_xyz_m,
                    forward_config.experiment.title,
                ),
                figure_directory,
                observation_index,
            )
        )

    animation_path = write_animation(frame_paths, figure_directory)
    metrics_path = configuration.data.output.metrics_path
    write_metrics_document(
        {
            "experiment_name": forward_config.experiment.name,
            "experiment_title": forward_config.experiment.title,
            "configuration_directory": str(arguments.configs),
            "generated_at": datetime.now(UTC).isoformat(),
            "spread_engine_name": forward_config.fire.spread_engine_name,
            "emission_model": forward_config.fire.emission_model,
            "wind_speed_m_per_s": forward_config.wind.wind_speed_m_per_s,
            "time_step_s": time_step_s,
            "observation_interval_s": forward_config.acoustic.observation_interval_s,
            "sample_rate_hz": forward_config.acoustic.sample_rate_hz,
            "receiver_count": int(receiver_positions_xyz_m.shape[0]),
            "receiver_positions_xyz_m": receiver_positions_xyz_m.tolist(),
            "observations": records,
        },
        metrics_path,
    )
    print(f"\nframes    {figure_directory}")
    print(f"animation {animation_path}")
    print(f"metrics   {metrics_path}")


if __name__ == "__main__":
    main()
