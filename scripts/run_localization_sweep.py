"""Sweeps one multi-source scene over the settings that decide whether it works.

Usage:
    uv run python scripts/run_localization_sweep.py --configs configs/m1
    uv run python scripts/run_localization_sweep.py --configs configs/m1 \
        --receiver-counts 3 4 6 --signal-to-noise-ratios-db 0 10 20

Zero domain logic lives here. Each cell of the grid is one whole run of
`run_multi_source_localization`, differing only in the settings named on the
command line, and each writes one metrics record. The deliverable is the
two-source resolution limit as a function of separation and receiver count.

This is the one script that imports another: reusing that script's run driver
keeps a single definition of what a run is, so a sweep cell and a single run
cannot drift apart. Nothing in `src/` imports either of them.
"""

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.run_multi_source_localization import (
    execute_run,
    stack_estimated_positions_xy_m,
    stack_refined_positions_xy_m,
)
from src.config.multi_source_localization_configuration import (
    PRODUCT_COMBINATOR,
    SUM_COMBINATOR,
)
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    SimulationConfiguration,
    load_simulation_configuration,
)
from src.config.source_scene_configuration import (
    SourceConfiguration,
    stack_source_positions_xy_m,
)
from src.utils.io.metrics_writer import append_metrics_record
from src.utils.metrics.localization_metrics import (
    build_localization_metric_record,
    compute_optimal_subpattern_assignment_distance,
)
from src.utils.visualization.localization_error_plotter import (
    plot_metric_against_parameter,
)

SWEEP_METRICS_FILENAME: str = "multi_source_localization_sweep.jsonl"
SWEEP_FIGURE_NAME: str = "localization_error_against_separation"


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
        "--source-separations-m", type=float, nargs="+", default=[10.0, 20.0, 40.0]
    )
    parser.add_argument(
        "--phase-transform-exponents", type=float, nargs="+", default=[0.7]
    )
    parser.add_argument(
        "--pairwise-combinators",
        type=str,
        nargs="+",
        default=[PRODUCT_COMBINATOR, SUM_COMBINATOR],
    )
    return parser.parse_args()


def place_two_sources_at_separation(
    configuration: SimulationConfiguration, separation_m: float
) -> tuple[SourceConfiguration, ...]:
    """Puts two sources either side of the domain centre, a set distance apart.

    Amplitude scales are carried over from the scene, so a sweep over separation
    keeps whatever power disparity the scene declared.

    Args:
        configuration: The whole run configuration.
        separation_m: Distance between the two sources, in metres.

    Returns:
        The two sources.

    Raises:
        ValueError: If the scene does not declare exactly two sources.
    """
    sources = configuration.geometry.concurrent_sources
    if len(sources) != 2:
        raise ValueError(
            "the separation sweep is defined for a scene with exactly two sources"
        )
    centre_xy_m = np.array(
        [
            0.5 * configuration.geometry.domain.size_x_m,
            0.5 * configuration.geometry.domain.size_y_m,
        ],
        dtype=np.float64,
    )
    offset_xy_m = np.array([0.5 * separation_m, 0.0], dtype=np.float64)
    return (
        replace(sources[0], position_xy_m=centre_xy_m - offset_xy_m),
        replace(sources[1], position_xy_m=centre_xy_m + offset_xy_m),
    )


def build_sweep_configuration(
    configuration: SimulationConfiguration,
    receiver_count: int,
    signal_to_noise_ratio_db: float,
    separation_m: float,
    phase_transform_exponent: float,
    pairwise_combinator: str,
) -> SimulationConfiguration:
    """Applies one cell of the sweep grid to the base configuration.

    Args:
        configuration: The base configuration.
        receiver_count: Receivers to place.
        signal_to_noise_ratio_db: Receiver noise level.
        separation_m: Distance between the two sources, in metres.
        phase_transform_exponent: Whitening exponent.
        pairwise_combinator: How pairwise maps are merged.

    Returns:
        The configuration for this cell.
    """
    multi_source = configuration.localization.multi_source
    return replace(
        configuration,
        geometry=replace(
            configuration.geometry,
            receiver_layout=replace(
                configuration.geometry.receiver_layout, count=receiver_count
            ),
            concurrent_sources=place_two_sources_at_separation(
                configuration, separation_m
            ),
        ),
        forward_model=replace(
            configuration.forward_model,
            receiver_noise=replace(
                configuration.forward_model.receiver_noise,
                enabled=True,
                signal_to_noise_ratio_db=signal_to_noise_ratio_db,
            ),
        ),
        localization=replace(
            configuration.localization,
            multi_source=replace(
                multi_source,
                correlation=replace(
                    multi_source.correlation,
                    phase_transform_exponent=phase_transform_exponent,
                ),
                steered_response_power=replace(
                    multi_source.steered_response_power,
                    pairwise_combinator=pairwise_combinator,
                ),
                clustering=replace(multi_source.clustering, enabled=False),
            ),
        ),
    )


def main() -> None:
    """Runs every cell of the grid and writes one metrics record per cell."""
    arguments = parse_arguments()
    base_configuration = load_simulation_configuration(arguments.configs)
    metrics_path = (
        base_configuration.data.output.metrics_directory / SWEEP_METRICS_FILENAME
    )
    metrics_configuration = base_configuration.localization.multi_source.metrics

    grid = list(
        itertools.product(
            arguments.receiver_counts,
            arguments.signal_to_noise_ratios_db,
            arguments.source_separations_m,
            arguments.phase_transform_exponents,
            arguments.pairwise_combinators,
        )
    )
    print(f"sweeping {len(grid)} cells into {metrics_path}")
    records: list[dict[str, object]] = []
    distances_m: list[float] = []
    for cell_index, cell in enumerate(grid):
        (
            receiver_count,
            signal_to_noise_ratio_db,
            separation_m,
            phase_transform_exponent,
            pairwise_combinator,
        ) = cell
        configuration = build_sweep_configuration(base_configuration, *cell)
        run = execute_run(configuration)
        true_positions_xy_m = stack_source_positions_xy_m(
            configuration.geometry.concurrent_sources
        )
        delay_only_distance_m = compute_optimal_subpattern_assignment_distance(
            true_positions_xy_m,
            stack_estimated_positions_xy_m(run),
            metrics_configuration.optimal_subpattern_assignment_cutoff_m,
            metrics_configuration.optimal_subpattern_assignment_order,
        )
        distances_m.append(delay_only_distance_m)
        record = {
            "cell_index": cell_index,
            "receiver_count": receiver_count,
            "signal_to_noise_ratio_db": signal_to_noise_ratio_db,
            "source_separation_m": separation_m,
            "phase_transform_exponent": phase_transform_exponent,
            "pairwise_combinator": pairwise_combinator,
            "stop_reason": run.localization.stop_reason,
            "estimated_source_count": run.localization.estimated_source_count,
            "delay_only_metrics": build_localization_metric_record(
                true_positions_xy_m,
                stack_estimated_positions_xy_m(run),
                metrics_configuration.optimal_subpattern_assignment_cutoff_m,
                metrics_configuration.optimal_subpattern_assignment_order,
            ),
            "level_fused_metrics": build_localization_metric_record(
                true_positions_xy_m,
                stack_refined_positions_xy_m(run),
                metrics_configuration.optimal_subpattern_assignment_cutoff_m,
                metrics_configuration.optimal_subpattern_assignment_order,
            ),
        }
        records.append(record)
        append_metrics_record(record, metrics_path)
        print(
            f"  [{cell_index + 1}/{len(grid)}] N={receiver_count} "
            f"SNR={signal_to_noise_ratio_db:.0f} dB sep={separation_m:.0f} m "
            f"beta={phase_transform_exponent} {pairwise_combinator}: "
            f"K_hat={run.localization.estimated_source_count} "
            f"OSPA={delay_only_distance_m:.2f} m"
        )

    separations_m = np.array(sorted(set(arguments.source_separations_m)))
    series = np.array(
        [
            [
                float(
                    np.mean(
                        [
                            distance_m
                            for record, distance_m in zip(
                                records, distances_m, strict=True
                            )
                            if record["source_separation_m"] == separation_m
                            and record["pairwise_combinator"] == combinator
                        ]
                        or [np.nan]
                    )
                )
                for separation_m in separations_m
            ]
            for combinator in arguments.pairwise_combinators
        ]
    )
    figure_directory = Path("results/figures") / metrics_path.stem
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure_path = figure_directory / f"{SWEEP_FIGURE_NAME}.svg"
    plot_metric_against_parameter(
        separations_m,
        series,
        list(arguments.pairwise_combinators),
        "source separation (m)",
        "optimal sub-pattern assignment (m)",
        "two-source resolution against separation",
    ).savefig(figure_path, format="svg", bbox_inches="tight")
    print(f"figure written to {figure_path}")
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
