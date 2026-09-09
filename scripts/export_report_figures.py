"""Regenerates every report figure from a saved run and a metrics document.

Usage:
    uv run python scripts/export_report_figures.py
    uv run python scripts/export_report_figures.py --run 20260909T083158Z
    uv run python scripts/export_report_figures.py --metrics results/metrics/x.json

The run picked defaults to the newest one, and the scene replayed for the
front figure comes from that run's own `config.json`, so the two can never
disagree. `--configs` overrides the scene when you want a different one.

Figures land in three kinds of directory under `results/figures/`:

    general/          the fire model and the ladder side by side, no one scene
    <experiment>/     everything derived from the chosen run, e.g. e1/, e3/
    <metrics stem>/   everything derived from a localization metrics document

Each carries a `figure_metadata.json` holding the resolved configuration that
produced it, so an E1 figure can be checked against the E1 scene rather than
taken on trust.

Zero plotting logic lives here: this script reads a run through
`simulation_run_reader`, calls the plotters in `src/utils/visualization/`, and
writes what they return. SVG for Overleaf, GIF for the defense. The figure
names are stable, so re-running overwrites rather than accumulating.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure

from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    ForwardSimulationConfiguration,
    load_forward_simulation_configuration,
    load_localization_configuration,
)
from src.config.simulation_context_factory import build_simulation_context
from src.spark.acoustic.burning_cell_source_model import compute_fire_front_mask
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.terrain.square_grid_mesh import SquareGridMesh
from src.utils.array_types import BoolArray, Float64Array
from src.utils.io.simulation_run_reader import (
    SimulationRun,
    list_simulation_runs,
    read_simulation_run,
)
from src.utils.visualization.channel_plotter import plot_channel_panels
from src.utils.visualization.fire_state_plotter import plot_ignition_time_map
from src.utils.visualization.mesh_plotter import plot_scene_geometry_panels
from src.utils.visualization.rate_of_spread_plotter import plot_rate_of_spread_panels
from src.utils.visualization.receiver_signal_plotter import (
    plot_error_against_bisector_distance,
    plot_estimates_against_truth,
    plot_front_position_over_observations,
    plot_receiver_level_traces,
    plot_reduced_chi_square_by_scenario,
)

RUN_ROOT: Path = Path("results/simulations")
FIGURE_ROOT: Path = Path("results/figures")
GENERAL_FIGURE_DIRECTORY_NAME: str = "general"
FIGURE_METADATA_FILENAME: str = "figure_metadata.json"
DEFAULT_METRICS_PATH: Path = Path("results/metrics/single_source_localization.json")
LADDER_DIRECTORIES: tuple[Path, ...] = (
    Path("configs/e1"),
    Path("configs/e2"),
    Path("configs/e3"),
    Path("configs/e4"),
)
LADDER_TITLES: tuple[str, ...] = ("E1", "E2", "E3", "E4")

WIND_SPEED_SWEEP_M_PER_S: Float64Array = np.linspace(0.0, 12.0, 120)
SLOPE_ANGLES_RAD: Float64Array = np.radians(np.array([0.0, 10.0, 20.0, 30.0]))
MOISTURE_SWEEP_FRACTION: Float64Array = np.linspace(0.02, 0.30, 60)
ABSORPTION_FREQUENCY_SWEEP_HZ: Float64Array = np.geomspace(50.0, 20000.0, 300)
OCTAVE_BAND_CENTRES_HZ: Float64Array = np.array(
    [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
)
CHANNEL_RANGE_SWEEP_M: Float64Array = np.linspace(1.0, 150.0, 150)

LEVEL_FLOOR: float = 1e-20
DECIBELS_PER_POWER_DECADE: float = 10.0
FRONT_SNAPSHOT_COUNT: int = 4
ANIMATION_FRAME_INTERVAL_MS: int = 300
ANIMATION_FRAMES_PER_SECOND: int = 4


def parse_arguments() -> argparse.Namespace:
    """Read the run identifier, metrics path and configuration directory.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--configs", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=FIGURE_ROOT)
    return parser.parse_args()


def resolve_run_directory(run_id: str | None) -> Path:
    """Pick the run to export, defaulting to the newest one.

    Args:
        run_id: Explicit run identifier, or None for the newest.

    Returns:
        Path to the run directory.

    Raises:
        FileNotFoundError: If no run exists under the run root.
    """
    if run_id is not None:
        return RUN_ROOT / run_id
    available = list_simulation_runs(RUN_ROOT)
    if not available:
        raise FileNotFoundError(f"no simulation runs found under {RUN_ROOT}")
    return RUN_ROOT / available[0]


def save_figure(figure: Figure, figure_directory: Path, name: str) -> Path:
    """Write one figure as SVG under a stable name.

    Args:
        figure: The figure to save.
        figure_directory: Directory to write into.
        name: Stable figure name, without extension.

    Returns:
        The path written.
    """
    figure_directory.mkdir(parents=True, exist_ok=True)
    path = figure_directory / f"{name}.svg"
    figure.savefig(path, format="svg", bbox_inches="tight")
    return path


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


def compute_equivalent_front_radius_m(run: SimulationRun) -> Float64Array:
    """Radius of a disc with the same burnt area, per observation.

    Preferred over the mean front-cell radius for the rate-of-spread fit. The
    burning set is one cell thick and sparse, so which cells happen to be
    alight at a sampling instant wanders by metres; burnt area only ever grows,
    and its equivalent radius is monotone by construction.

    Args:
        run: The saved run.

    Returns:
        Float64 array of shape (n_observations,), in metres.
    """
    burnt_area_m2 = read_ground_truth_series(run, "burnt_area_m2")
    return np.asarray(np.sqrt(burnt_area_m2 / np.pi), dtype=np.float64)


def replay_front_masks(
    config: ForwardSimulationConfiguration,
) -> tuple[SquareGridMesh, Any, list[BoolArray], list[float]]:
    """Re-run the fire alone to recover front masks the run does not store.

    The run directory stores aggregates, not per-cell state, because the state
    is hundreds of megabytes and cheap to regenerate: the fire without any
    acoustics takes under a second.

    Args:
        config: The scene to replay, normally the one the run itself carries.

    Returns:
        The mesh, the final fire state, the front masks and their times.

    Raises:
        TypeError: If the configured mesh is not a square grid, which the
            ignition-time raster needs.
    """
    context = build_simulation_context(config)
    mesh = context.mesh
    if not isinstance(mesh, SquareGridMesh):
        raise TypeError("the ignition-time raster needs a square grid mesh")

    maximum_rate_of_spread_m_per_s = float(
        compute_rate_of_spread_balbi_2009(
            np.array([config.wind.wind_speed_m_per_s]), np.zeros(1), context.fuel
        )[0]
    )
    time_step_s = compute_maximum_stable_time_step_s(
        mesh,
        maximum_rate_of_spread_m_per_s,
        config.fire.time_step_safety_factor,
        context.fuel.residence_time_s,
    )
    state = context.spread_engine.initialize(
        mesh, context.fuel_field, context.wind_field
    )
    state = context.spread_engine.ignite_cells(state, context.ignition_cell_indices)

    step_count = max(1, int(config.fire.simulation_duration_s / time_step_s))
    snapshot_steps = {
        round(fraction * step_count)
        for fraction in np.linspace(0.25, 1.0, FRONT_SNAPSHOT_COUNT)
    }
    front_masks: list[BoolArray] = []
    front_mask_times_s: list[float] = []
    for step_index in range(1, step_count + 1):
        state = context.spread_engine.step(state, time_step_s)
        if step_index in snapshot_steps:
            front_masks.append(compute_fire_front_mask(state, mesh.neighbor_indices))
            front_mask_times_s.append(float(state.current_time_s))
    return mesh, state, front_masks, front_mask_times_s


def export_scene_figures(figure_directory: Path) -> list[Path]:
    """Export the ladder geometry panel, F2.

    Args:
        figure_directory: Directory to write into.

    Returns:
        The paths written.
    """
    titles: list[str] = []
    extents_xy_m: list[tuple[float, float]] = []
    ignition_positions_xy_m: list[Float64Array] = []
    receiver_positions_xy_m: list[Float64Array] = []
    for title, configuration_directory in zip(
        LADDER_TITLES, LADDER_DIRECTORIES, strict=True
    ):
        if not configuration_directory.is_dir():
            continue
        config = load_forward_simulation_configuration(configuration_directory)
        context = build_simulation_context(config)
        titles.append(title)
        extents_xy_m.append((config.mesh.extent_x_m, config.mesh.extent_y_m))
        ignition_positions_xy_m.append(
            context.mesh.cell_positions_xyz[context.ignition_cell_indices, :2]
        )
        receiver_positions_xy_m.append(context.receiver_positions_xy_m)

    if not titles:
        return []
    localization = load_localization_configuration(
        DEFAULT_CONFIGURATION_DIRECTORY / "localization.toml"
    )
    figure = plot_scene_geometry_panels(
        titles,
        extents_xy_m,
        ignition_positions_xy_m,
        receiver_positions_xy_m,
        localization.triangulation.near_singular_tolerance,
    )
    return [save_figure(figure, figure_directory, "f2_scene_geometry")]


def read_metrics_document(metrics_path: Path) -> dict[str, Any]:
    """Read one localization metrics document.

    Args:
        metrics_path: Path to the document.

    Returns:
        The parsed document.
    """
    document: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return document


def export_metrics_figures(metrics_path: Path, figure_directory: Path) -> list[Path]:
    """Export the estimator figures F6, F7 and F8 from a metrics document.

    Args:
        metrics_path: Path to a localization metrics document.
        figure_directory: Directory to write into.

    Returns:
        The paths written, empty when the metrics document is absent.
    """
    if not metrics_path.is_file():
        return []
    metrics = read_metrics_document(metrics_path)
    receiver_positions_xy_m = np.asarray(
        metrics["receiver_positions_xy_m"], dtype=np.float64
    )
    written: list[Path] = []

    scenario_labels: list[str] = []
    degenerate_scenarios: list[bool] = []
    reduced_chi_squares: list[Float64Array] = []
    bisector_distances_m: list[float] = []
    errors_m: list[float] = []
    midpoint_xy_m = receiver_positions_xy_m.mean(axis=0)
    baseline_xy_m = receiver_positions_xy_m[1] - receiver_positions_xy_m[0]
    baseline_direction = baseline_xy_m / np.linalg.norm(baseline_xy_m)

    for scenario_index, scenario in enumerate(metrics["scenarios"]):
        estimates = scenario["clip_estimates"]
        true_position_xy_m = np.asarray(
            scenario["true_position_xy_m"], dtype=np.float64
        )
        estimated_positions_xy_m = np.asarray(
            [estimate["position_xy_m"] for estimate in estimates], dtype=np.float64
        )
        figure = plot_estimates_against_truth(
            estimated_positions_xy_m,
            np.asarray(
                [estimate["ellipse_semi_major_m"] for estimate in estimates],
                dtype=np.float64,
            ),
            np.asarray(
                [estimate["ellipse_semi_minor_m"] for estimate in estimates],
                dtype=np.float64,
            ),
            true_position_xy_m,
            receiver_positions_xy_m,
            f"scenario {scenario_index + 1}",
        )
        written.append(
            save_figure(
                figure, figure_directory, f"f6_estimates_scenario_{scenario_index + 1}"
            )
        )

        scenario_labels.append(f"S{scenario_index + 1}")
        degenerate_scenarios.append(
            all(estimate["is_near_singular"] for estimate in estimates)
        )
        reduced_chi_squares.append(
            np.asarray(
                [estimate["reduced_chi_square"] for estimate in estimates],
                dtype=np.float64,
            )
        )
        bisector_distances_m.append(
            abs(float(np.dot(true_position_xy_m - midpoint_xy_m, baseline_direction)))
        )
        errors_m.append(float(scenario["median_error_m"]))

    written.append(
        save_figure(
            plot_error_against_bisector_distance(
                np.asarray(bisector_distances_m, dtype=np.float64),
                np.asarray(errors_m, dtype=np.float64),
                float(np.linalg.norm(baseline_xy_m)) * 0.05,
            ),
            figure_directory,
            "f7_error_against_bisector_distance",
        )
    )
    written.append(
        save_figure(
            plot_reduced_chi_square_by_scenario(
                scenario_labels, reduced_chi_squares, degenerate_scenarios
            ),
            figure_directory,
            "f8_reduced_chi_square",
        )
    )
    return written


def export_front_animation(run: SimulationRun, figure_directory: Path) -> Path:
    """Animate the front travelling with the receiver levels ticking alongside.

    Args:
        run: The saved run.
        figure_directory: Directory to write into.

    Returns:
        The path written.
    """
    observation_times_s = run.observation_times_s
    front_centroids_xy_m = np.asarray(
        [record["front_centroid_xy_m"] for record in run.ground_truth],
        dtype=np.float64,
    )
    receiver_levels_db = compute_receiver_levels_db(run)
    extent_x_m = float(run.config["mesh"]["extent_x_m"])
    extent_y_m = float(run.config["mesh"]["extent_y_m"])

    figure = Figure(figsize=(7.0, 3.2))
    scene_axes = figure.add_subplot(121)
    level_axes = figure.add_subplot(122)
    scene_axes.scatter(
        run.receiver_positions_xy_m[:, 0],
        run.receiver_positions_xy_m[:, 1],
        marker="v",
        s=30,
        color="#2f5d8a",
    )
    front_artist = scene_axes.scatter([], [], marker="*", s=110, color="#e08a1e")
    scene_axes.set_xlim(0.0, extent_x_m)
    scene_axes.set_ylim(0.0, extent_y_m)
    scene_axes.set_aspect("equal")
    scene_axes.set_xlabel("x (m)")
    scene_axes.set_ylabel("y (m)")

    for receiver_index in range(run.receiver_count):
        level_axes.plot(
            observation_times_s, receiver_levels_db[:, receiver_index], alpha=0.35
        )
    time_marker = level_axes.axvline(observation_times_s[0], color="#d94a3d")
    level_axes.set_xlabel("simulation time (s)")
    level_axes.set_ylabel("received level (dB)")

    def draw_observation(observation_index: int) -> tuple[Any, ...]:
        front_artist.set_offsets(front_centroids_xy_m[observation_index].reshape(1, 2))
        time_marker.set_xdata([observation_times_s[observation_index]] * 2)
        scene_axes.set_title(f"t = {observation_times_s[observation_index]:.0f} s")
        return front_artist, time_marker

    animation = FuncAnimation(
        figure,
        draw_observation,
        frames=run.observation_count,
        interval=ANIMATION_FRAME_INTERVAL_MS,
        blit=False,
    )
    figure_directory.mkdir(parents=True, exist_ok=True)
    path = figure_directory / "f3_front_evolution.gif"
    animation.save(path, writer=PillowWriter(fps=ANIMATION_FRAMES_PER_SECOND))
    return path


def write_figure_metadata(
    figure_directory: Path,
    written: list[Path],
    provenance: dict[str, Any],
) -> Path:
    """Record what produced the figures sitting in one directory.

    A figure on its own says nothing about which configuration made it. This
    sidecar carries the whole resolved configuration next to the figures, so a
    reader can check that an E1 figure really came from the E1 scene.

    Args:
        figure_directory: Directory the figures were written into.
        written: Paths of the figures written there.
        provenance: The run, the configuration and anything else that produced
            them.

    Returns:
        The path written.
    """
    figure_directory.mkdir(parents=True, exist_ok=True)
    path = figure_directory / FIGURE_METADATA_FILENAME
    document = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "generated_by": "scripts/export_report_figures.py",
        "figures": sorted(figure.name for figure in written),
        **provenance,
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> list[Path]:
    """Regenerate every figure the report needs.

    Figures land in three kinds of directory: `general/` for the ones that
    depend on no scene, `<experiment>/` for the ones derived from the chosen
    run, and `<metrics stem>/` for the ones derived from a metrics document.
    Each directory carries a `figure_metadata.json` recording exactly what
    produced it.

    Returns:
        The paths written.
    """
    arguments = parse_arguments()
    run_directory = resolve_run_directory(arguments.run)
    run = read_simulation_run(run_directory)
    scene_config = (
        load_forward_simulation_configuration(arguments.configs)
        if arguments.configs is not None
        else ForwardSimulationConfiguration.from_dict(run.config)
    )
    conditions = scene_config.atmosphere

    figure_root: Path = arguments.figures
    general_directory = figure_root / GENERAL_FIGURE_DIRECTORY_NAME
    experiment_directory = figure_root / scene_config.experiment.name
    written: list[Path] = []

    general_written = [
        save_figure(
            plot_rate_of_spread_panels(
                WIND_SPEED_SWEEP_M_PER_S,
                SLOPE_ANGLES_RAD,
                MOISTURE_SWEEP_FRACTION,
                FuelProperties.pine_needle_litter(),
            ),
            general_directory,
            "f1_rate_of_spread",
        ),
        *export_scene_figures(general_directory),
    ]
    write_figure_metadata(
        general_directory,
        general_written,
        {
            "scope": "general",
            "note": (
                "Figures that depend on no single scene: the fire model itself, "
                "and every rung of the ladder side by side."
            ),
            "ladder_directories": [
                str(directory) for directory in LADDER_DIRECTORIES if directory.is_dir()
            ],
        },
    )
    written.extend(general_written)

    mesh, final_state, front_masks, front_mask_times_s = replay_front_masks(
        scene_config
    )
    experiment_written = [
        save_figure(
            plot_ignition_time_map(final_state, mesh, front_masks, front_mask_times_s),
            experiment_directory,
            "f3_front_evolution",
        ),
        save_figure(
            plot_channel_panels(
                ABSORPTION_FREQUENCY_SWEEP_HZ,
                OCTAVE_BAND_CENTRES_HZ,
                CHANNEL_RANGE_SWEEP_M,
                conditions,
                scene_config.acoustic.reference_distance_m,
            ),
            experiment_directory,
            "f4_channel",
        ),
        save_figure(
            plot_receiver_level_traces(
                run.observation_times_s,
                compute_receiver_levels_db(run),
                run.receiver_positions_xy_m,
            ),
            experiment_directory,
            "f5_receiver_levels",
        ),
        save_figure(
            plot_front_position_over_observations(
                run.observation_times_s,
                compute_equivalent_front_radius_m(run),
                None,
            ),
            experiment_directory,
            "f9_front_position",
        ),
        export_front_animation(run, experiment_directory),
    ]
    write_figure_metadata(
        experiment_directory,
        experiment_written,
        {
            "scope": "experiment",
            "experiment": scene_config.experiment.to_dict(),
            "run_id": run.run_id,
            "run_directory": run_directory.as_posix(),
            "run_metadata": run.metadata,
            "configuration": scene_config.to_dict(),
        },
    )
    written.extend(experiment_written)

    if arguments.metrics.is_file():
        metrics_directory = figure_root / arguments.metrics.stem
        metrics_written = export_metrics_figures(arguments.metrics, metrics_directory)
        write_figure_metadata(
            metrics_directory,
            metrics_written,
            {
                "scope": "metrics",
                "note": (
                    "Estimator figures. These come from a localization metrics "
                    "document, not from the forward run above."
                ),
                "metrics_path": arguments.metrics.as_posix(),
                "metrics_settings": {
                    key: value
                    for key, value in read_metrics_document(arguments.metrics).items()
                    if key != "scenarios"
                },
            },
        )
        written.extend(metrics_written)

    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main()
