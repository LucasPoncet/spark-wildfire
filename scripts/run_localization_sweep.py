"""Sweeps one multi-source scene over the settings that decide whether it works.

Usage:
    uv run python scripts/run_localization_sweep.py --configs configs/m1
    uv run python scripts/run_localization_sweep.py --configs configs/m3 \
        --receiver-counts 3 4 5 6 8 --source-counts 2 3 4

Zero domain logic lives here. Each cell of the grid is one whole run of
`run_multi_source_localization`, differing only in the settings named on the
command line, and each writes one metrics record. The deliverables are accuracy
against receiver count at each source count, and the two-source resolution limit
against separation.

This is the one script that imports another: reusing that script's run driver
keeps a single definition of what a run is, so a sweep cell and a single run
cannot drift apart. Nothing in `src/` imports either of them.
"""

import argparse
import itertools
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from scripts.run_multi_source_localization import (
    compute_radial_semi_axis_m,
    execute_run,
    stack_estimated_positions_xy_m,
    stack_refined_positions_xy_m,
)
from src.config.multi_source_localization_configuration import PRODUCT_COMBINATOR
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    SimulationConfiguration,
    load_simulation_configuration,
)
from src.config.source_scene_configuration import (
    SourceConfiguration,
    stack_source_positions_xy_m,
)
from src.utils.array_types import Float64Array
from src.utils.io.metrics_writer import append_metrics_record
from src.utils.metrics.localization_metrics import (
    build_localization_metric_record,
    compute_optimal_subpattern_assignment_distance,
)
from src.utils.visualization.localization_error_plotter import (
    plot_metric_against_parameter,
    plot_source_count_confusion,
)
from src.utils.visualization.steered_response_power_plotter import (
    plot_steered_response_power_map,
)

SWEEP_METRICS_SUFFIX: str = "_sweep.jsonl"
FIGURE_ROOT: Path = Path("results/figures")
FIGURE_FORMAT: str = "svg"
PAIR_SOURCE_COUNT: int = 2


@dataclass(frozen=True)
class SweepCell:
    """One point of the sweep grid.

    Attributes:
        receiver_count: Receivers to place.
        signal_to_noise_ratio_db: Receiver noise level.
        source_count: How many of the scene's sources to sound.
        source_separation_m: Distance between the two sources, or None to leave
            them where the scene puts them.
        phase_transform_exponent: Whitening exponent.
        pairwise_combinator: How pairwise maps are merged.
    """

    receiver_count: int
    signal_to_noise_ratio_db: float
    source_count: int
    source_separation_m: float | None
    phase_transform_exponent: float
    pairwise_combinator: str


@dataclass(frozen=True)
class SweepOutcome:
    """What one cell produced, in the form the figures need.

    Attributes:
        cell: The cell that produced it.
        estimated_source_count: How many sources the loop accepted.
        optimal_subpattern_assignment_m: Delay-only score against the truth.
        median_ellipse_semi_major_m: Median stated precision over the sources
            that were found, or not-a-number when none were.
        record: The whole JSON-serializable record.
    """

    cell: SweepCell
    estimated_source_count: int
    optimal_subpattern_assignment_m: float
    median_ellipse_semi_major_m: float
    record: dict[str, object]


def parse_arguments() -> argparse.Namespace:
    """Parses the sweep grid from the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    parser.add_argument("--receiver-counts", type=int, nargs="+", default=[4, 6])
    parser.add_argument(
        "--signal-to-noise-ratios-db", type=float, nargs="+", default=[20.0]
    )
    parser.add_argument(
        "--source-counts",
        type=int,
        nargs="+",
        default=[],
        help="how many of the scene's sources to sound; empty means all of them",
    )
    parser.add_argument(
        "--source-separations-m",
        type=float,
        nargs="+",
        default=[],
        help="two-source cells only; empty leaves the scene's own positions",
    )
    parser.add_argument(
        "--phase-transform-exponents", type=float, nargs="+", default=[0.7]
    )
    parser.add_argument(
        "--pairwise-combinators", type=str, nargs="+", default=[PRODUCT_COMBINATOR]
    )
    return parser.parse_args()


def build_sweep_grid(
    arguments: argparse.Namespace, scene_source_count: int
) -> list[SweepCell]:
    """Expands the command line into one cell per combination.

    Args:
        arguments: The parsed arguments.
        scene_source_count: How many sources the scene declares.

    Returns:
        The cells, in the order they will be run.
    """
    source_counts = arguments.source_counts or [scene_source_count]
    separations_m: list[float | None] = list(arguments.source_separations_m) or [None]
    return [
        SweepCell(
            receiver_count=receiver_count,
            signal_to_noise_ratio_db=signal_to_noise_ratio_db,
            source_count=source_count,
            source_separation_m=separation_m,
            phase_transform_exponent=phase_transform_exponent,
            pairwise_combinator=pairwise_combinator,
        )
        for (
            receiver_count,
            signal_to_noise_ratio_db,
            source_count,
            separation_m,
            phase_transform_exponent,
            pairwise_combinator,
        ) in itertools.product(
            arguments.receiver_counts,
            arguments.signal_to_noise_ratios_db,
            source_counts,
            separations_m,
            arguments.phase_transform_exponents,
            arguments.pairwise_combinators,
        )
    ]


def place_two_sources_at_separation(
    sources: tuple[SourceConfiguration, ...],
    domain_extent_x_m: float,
    domain_extent_y_m: float,
    separation_m: float,
) -> tuple[SourceConfiguration, ...]:
    """Puts two sources either side of the domain centre, a set distance apart.

    Amplitude scales are carried over from the scene, so a sweep over separation
    keeps whatever power disparity the scene declared.

    Args:
        sources: The scene's sources, of which there must be exactly two.
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        separation_m: Distance between the two sources, in metres.

    Returns:
        The two sources.

    Raises:
        ValueError: If the cell does not offer exactly two sources.
    """
    if len(sources) != PAIR_SOURCE_COUNT:
        raise ValueError(
            "a separation sweep is defined for two sources only; this cell has "
            f"{len(sources)}"
        )
    centre_xy_m = np.array(
        [0.5 * domain_extent_x_m, 0.5 * domain_extent_y_m], dtype=np.float64
    )
    offset_xy_m = np.array([0.5 * separation_m, 0.0], dtype=np.float64)
    return (
        replace(sources[0], position_xy_m=centre_xy_m - offset_xy_m),
        replace(sources[1], position_xy_m=centre_xy_m + offset_xy_m),
    )


def select_scene_sources(
    configuration: SimulationConfiguration, cell: SweepCell
) -> tuple[SourceConfiguration, ...]:
    """Takes the cell's source count out of the scene, moving them if asked.

    Args:
        configuration: The base configuration.
        cell: The sweep cell.

    Returns:
        The sources this cell sounds.

    Raises:
        ValueError: If the scene declares fewer sources than the cell wants.
    """
    scene_sources = configuration.geometry.concurrent_sources
    if cell.source_count > len(scene_sources):
        raise ValueError(
            f"the scene declares {len(scene_sources)} sources, fewer than the "
            f"{cell.source_count} this cell asks for"
        )
    sources = scene_sources[: cell.source_count]
    if cell.source_separation_m is None:
        return sources
    return place_two_sources_at_separation(
        sources,
        configuration.geometry.domain.size_x_m,
        configuration.geometry.domain.size_y_m,
        cell.source_separation_m,
    )


def build_sweep_configuration(
    configuration: SimulationConfiguration, cell: SweepCell
) -> SimulationConfiguration:
    """Applies one cell of the sweep grid to the base configuration.

    Args:
        configuration: The base configuration.
        cell: The sweep cell.

    Returns:
        The configuration for this cell.
    """
    multi_source = configuration.localization.multi_source
    return replace(
        configuration,
        geometry=replace(
            configuration.geometry,
            receiver_layout=replace(
                configuration.geometry.receiver_layout, count=cell.receiver_count
            ),
            concurrent_sources=select_scene_sources(configuration, cell),
        ),
        forward_model=replace(
            configuration.forward_model,
            receiver_noise=replace(
                configuration.forward_model.receiver_noise,
                enabled=True,
                signal_to_noise_ratio_db=cell.signal_to_noise_ratio_db,
            ),
        ),
        localization=replace(
            configuration.localization,
            multi_source=replace(
                multi_source,
                correlation=replace(
                    multi_source.correlation,
                    phase_transform_exponent=cell.phase_transform_exponent,
                ),
                steered_response_power=replace(
                    multi_source.steered_response_power,
                    pairwise_combinator=cell.pairwise_combinator,
                ),
                clustering=replace(multi_source.clustering, enabled=False),
            ),
        ),
    )


def describe_cell(cell: SweepCell) -> str:
    """Names one cell so its figure file says which run produced it.

    Every swept axis goes into the name, not only the ones this particular grid
    varies, so a figure stays readable away from the command line that made it.

    Args:
        cell: The sweep cell.

    Returns:
        A filename-safe slug.
    """
    separation = (
        "sepscene"
        if cell.source_separation_m is None
        else f"sep{cell.source_separation_m:g}m"
    )
    return (
        f"n{cell.receiver_count:02d}_k{cell.source_count:02d}_"
        f"snr{cell.signal_to_noise_ratio_db:g}db_{separation}_"
        f"beta{cell.phase_transform_exponent:g}_{cell.pairwise_combinator}"
    )


def run_cell(
    base_configuration: SimulationConfiguration,
    cell: SweepCell,
    figure_directory: Path,
) -> SweepOutcome:
    """Runs one whole localization at this cell's settings and scores it.

    Args:
        base_configuration: The scene the sweep varies.
        cell: The sweep cell.
        figure_directory: Directory to write this cell's search map into.

    Returns:
        The outcome, ready to record and plot.
    """
    configuration = build_sweep_configuration(base_configuration, cell)
    metrics_configuration = configuration.localization.multi_source.metrics
    run = execute_run(configuration)
    true_positions_xy_m = stack_source_positions_xy_m(
        configuration.geometry.concurrent_sources
    )
    estimated_positions_xy_m = stack_estimated_positions_xy_m(run)
    semi_axes_m = [
        compute_radial_semi_axis_m(source.position.position_covariance_m2)
        for source in run.localization.sources
    ]
    save_figure(
        figure_directory,
        f"steered_response_power_map_{describe_cell(cell)}",
        plot_steered_response_power_map(
            run.localization.first_round_map,
            run.localization.first_round_positions_xyz_m,
            run.receiver_positions_xyz_m,
            true_positions_xy_m,
            estimated_positions_xy_m,
            [
                source.position.position_covariance_m2
                for source in run.localization.sources
            ],
            f"N = {cell.receiver_count}, K = {cell.source_count}, "
            f"K_hat = {run.localization.estimated_source_count}",
        ),
    )
    return SweepOutcome(
        cell=cell,
        estimated_source_count=run.localization.estimated_source_count,
        optimal_subpattern_assignment_m=(
            compute_optimal_subpattern_assignment_distance(
                true_positions_xy_m,
                estimated_positions_xy_m,
                metrics_configuration.optimal_subpattern_assignment_cutoff_m,
                metrics_configuration.optimal_subpattern_assignment_order,
            )
        ),
        median_ellipse_semi_major_m=(
            float(np.median(semi_axes_m)) if semi_axes_m else float("nan")
        ),
        record={
            "receiver_count": cell.receiver_count,
            "signal_to_noise_ratio_db": cell.signal_to_noise_ratio_db,
            "source_count": cell.source_count,
            "source_separation_m": cell.source_separation_m,
            "phase_transform_exponent": cell.phase_transform_exponent,
            "pairwise_combinator": cell.pairwise_combinator,
            "achieved_excerpt_coherence": run.achieved_excerpt_coherence,
            "source_recordings": [
                excerpt.recording_path.name for excerpt in run.excerpts
            ],
            "source_excerpt_offsets_s": [
                excerpt.start_sample_index / excerpt.sample_rate_hz
                for excerpt in run.excerpts
            ],
            "stop_reason": run.localization.stop_reason,
            "estimated_source_count": run.localization.estimated_source_count,
            "true_source_positions_xy_m": true_positions_xy_m,
            "estimated_positions_xy_m": estimated_positions_xy_m,
            "delay_only_ellipse_semi_major_m": semi_axes_m,
            "delay_only_metrics": build_localization_metric_record(
                true_positions_xy_m,
                estimated_positions_xy_m,
                metrics_configuration.optimal_subpattern_assignment_cutoff_m,
                metrics_configuration.optimal_subpattern_assignment_order,
            ),
            "level_fused_metrics": build_localization_metric_record(
                true_positions_xy_m,
                stack_refined_positions_xy_m(run),
                metrics_configuration.optimal_subpattern_assignment_cutoff_m,
                metrics_configuration.optimal_subpattern_assignment_order,
            ),
        },
    )


def summarise_over(
    outcomes: list[SweepOutcome],
    parameter_values: Float64Array,
    parameter_field: str,
    series_field: str,
    series_values: list[object],
    metric_field: str,
) -> Float64Array:
    """Averages one metric over every cell sharing a parameter and a series.

    Args:
        outcomes: Every cell that ran.
        parameter_values: The x-axis values.
        parameter_field: Name of the `SweepCell` field on the x-axis.
        series_field: Name of the `SweepCell` field separating the lines.
        series_values: One value per line.
        metric_field: Name of the `SweepOutcome` field to average.

    Returns:
        Array of shape `(n_series, n_parameter_values)`, not-a-number where no
        cell matched.
    """
    return np.array(
        [
            [
                float(
                    np.mean(
                        [
                            getattr(outcome, metric_field)
                            for outcome in outcomes
                            if getattr(outcome.cell, parameter_field) == parameter_value
                            and getattr(outcome.cell, series_field) == series_value
                        ]
                        or [np.nan]
                    )
                )
                for parameter_value in parameter_values
            ]
            for series_value in series_values
        ]
    )


def save_figure(figure_directory: Path, name: str, figure: Figure) -> Path:
    """Writes one figure under the sweep's figure directory.

    Args:
        figure_directory: Directory to write into, created if absent.
        name: File stem.
        figure: The figure to write.

    Returns:
        The path written to.
    """
    figure_directory.mkdir(parents=True, exist_ok=True)
    path = figure_directory / f"{name}.{FIGURE_FORMAT}"
    figure.savefig(path, format=FIGURE_FORMAT, bbox_inches="tight")
    return path


def export_sweep_figures(
    outcomes: list[SweepOutcome], figure_directory: Path
) -> list[Path]:
    """Draws whichever curves the swept grid actually supports.

    Args:
        outcomes: Every cell that ran.
        figure_directory: Directory to write into.

    Returns:
        The paths written.
    """
    receiver_counts = sorted({outcome.cell.receiver_count for outcome in outcomes})
    source_counts = sorted({outcome.cell.source_count for outcome in outcomes})
    separations_m = sorted(
        {
            outcome.cell.source_separation_m
            for outcome in outcomes
            if outcome.cell.source_separation_m is not None
        }
    )
    combinators = sorted({outcome.cell.pairwise_combinator for outcome in outcomes})
    source_count_labels = [f"K = {source_count}" for source_count in source_counts]
    paths: list[Path] = []

    if len(receiver_counts) > 1:
        receiver_axis = np.array(receiver_counts, dtype=np.float64)
        paths.append(
            save_figure(
                figure_directory,
                "localization_error_against_receiver_count",
                plot_metric_against_parameter(
                    receiver_axis,
                    summarise_over(
                        outcomes,
                        receiver_axis,
                        "receiver_count",
                        "source_count",
                        list(source_counts),
                        "optimal_subpattern_assignment_m",
                    ),
                    source_count_labels,
                    "receiver count",
                    "optimal sub-pattern assignment (m)",
                    "accuracy against receiver count",
                ),
            )
        )
        paths.append(
            save_figure(
                figure_directory,
                "stated_precision_against_receiver_count",
                plot_metric_against_parameter(
                    receiver_axis,
                    summarise_over(
                        outcomes,
                        receiver_axis,
                        "receiver_count",
                        "source_count",
                        list(source_counts),
                        "median_ellipse_semi_major_m",
                    ),
                    source_count_labels,
                    "receiver count",
                    "median error ellipse semi-major axis (m)",
                    "stated precision against receiver count",
                ),
            )
        )

    if len(separations_m) > 1:
        separation_axis = np.array(separations_m, dtype=np.float64)
        separation_series_values: list[object]
        if len(receiver_counts) > 1:
            separation_series_field = "receiver_count"
            separation_series_values = list(receiver_counts)
            separation_labels = [f"N = {count}" for count in receiver_counts]
        else:
            separation_series_field = "pairwise_combinator"
            separation_series_values = list(combinators)
            separation_labels = list(combinators)
        paths.append(
            save_figure(
                figure_directory,
                "localization_error_against_separation",
                plot_metric_against_parameter(
                    separation_axis,
                    summarise_over(
                        outcomes,
                        separation_axis,
                        "source_separation_m",
                        separation_series_field,
                        separation_series_values,
                        "optimal_subpattern_assignment_m",
                    ),
                    separation_labels,
                    "source separation (m)",
                    "optimal sub-pattern assignment (m)",
                    "two-source resolution against separation",
                ),
            )
        )

    paths.append(
        save_figure(
            figure_directory,
            "source_count_confusion",
            plot_source_count_confusion(
                np.array(
                    [outcome.cell.source_count for outcome in outcomes],
                    dtype=np.float64,
                ),
                np.array(
                    [outcome.estimated_source_count for outcome in outcomes],
                    dtype=np.float64,
                ),
                "estimated against true source count",
            ),
        )
    )
    return paths


def main() -> None:
    """Runs every cell of the grid and writes one metrics record per cell."""
    arguments = parse_arguments()
    base_configuration = load_simulation_configuration(arguments.configs)
    output = base_configuration.data.output
    metrics_path = output.metrics_directory / (
        Path(output.metrics_filename).stem + SWEEP_METRICS_SUFFIX
    )
    grid = build_sweep_grid(
        arguments, len(base_configuration.geometry.concurrent_sources)
    )
    figure_directory = FIGURE_ROOT / metrics_path.stem
    print(f"sweeping {len(grid)} cells into {metrics_path}")

    outcomes: list[SweepOutcome] = []
    for cell_index, cell in enumerate(grid):
        outcome = run_cell(base_configuration, cell, figure_directory)
        outcomes.append(outcome)
        append_metrics_record(
            {"cell_index": cell_index, **outcome.record}, metrics_path
        )
        separation = (
            "as configured"
            if cell.source_separation_m is None
            else f"{cell.source_separation_m:.0f} m apart"
        )
        print(
            f"  [{cell_index + 1}/{len(grid)}] N={cell.receiver_count} "
            f"K={cell.source_count} SNR={cell.signal_to_noise_ratio_db:.0f} dB "
            f"{separation} beta={cell.phase_transform_exponent} "
            f"{cell.pairwise_combinator}: "
            f"K_hat={outcome.estimated_source_count} "
            f"OSPA={outcome.optimal_subpattern_assignment_m:.3f} m "
            f"ellipse={outcome.median_ellipse_semi_major_m:.3f} m"
        )

    for path in export_sweep_figures(outcomes, figure_directory):
        print(f"figure written to {path}")
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
