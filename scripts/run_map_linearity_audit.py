"""Measures whether a steered response power map adds the way imaging assumes.

Usage:
    uv run python scripts/run_map_linearity_audit.py --configs configs/f1

Every moment, deconvolution and model fit downstream of this treats the map as
a source density convolved with one fixed response:

    map(u) ~= sum over sources k of  power_k * response(u; u_k)

That holds exactly for a plain cross-correlation summed over pairs. It does not
hold for the phase transform, which divides by the magnitude of the *mixture*
and is therefore not separable across sources, and it does not hold for the
product combinator, which is deliberately non-additive so that every pair must
agree before a cell scores.

So the audit renders each source alone, renders them together, and reports how
far the joint map is from the sum of the individual ones. It sweeps the source
count, the phase transform exponent, the combinator and the pooling rule, and
the configuration it selects is written into `[imaging_map]`.

The sweep runs on a coarser grid than the imaging map does, purely for cost.
That shortcut is checked rather than asserted: the chosen family is re-measured
at the imaging spacing and both residuals are reported. They are not equal —
the finer grid resolves more of the structure the two maps disagree about — so
the coarse sweep ranks families and the fine measurement is the one a gate is
read against.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from src.config.fire_characterization_configuration import (
    FIRE_CHARACTERIZATION_FILENAME,
    load_fire_characterization_configuration,
)
from src.config.multi_source_localization_configuration import (
    CorrelationConfiguration,
)
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    SimulationConfiguration,
    load_simulation_configuration,
)
from src.spark.acoustic.burning_cell_source_model import (
    generate_source_signal_for_burning_cell,
)
from src.spark.acoustic.free_field_propagation import (
    render_multi_source_receiver_signals,
)
from src.spark.acoustic.receiver_placement import place_receivers_from_layout
from src.spark.atmosphere.atmospheric_conditions import compute_speed_of_sound_m_per_s
from src.spark.inverse.map_normalization import normalize_map_to_unit_mass
from src.spark.inverse.receiver_pair_index import (
    compute_maximum_absolute_lag_s,
    enumerate_receiver_pairs,
)
from src.spark.inverse.steered_response_power import (
    build_candidate_grid_xyz,
    combine_pairwise_maps,
    compute_cell_delay_bounds_s,
    pool_curve_over_intervals,
)
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
    compute_pair_correlation_curves,
)
from src.utils.array_types import Float64Array
from src.utils.io.metrics_writer import write_metrics_document
from src.utils.visualization.map_linearity_plotter import plot_map_linearity_audit

FIGURE_ROOT: Path = Path("results/figures")
METRICS_ROOT: Path = Path("results/metrics")
SOURCE_RING_RADIUS_M: float = 5.0
SOURCE_SIGNAL_SEED: int = 4096
AUDIT_SEGMENT_DURATION_S: float = 4.0
LINEARITY_CORRELATION_TARGET: float = 0.98
LINEARITY_RESIDUAL_TARGET: float = 0.05
GATE_SOURCE_COUNT: int = 8


@dataclass(frozen=True)
class LinearityCell:
    """One swept combination and what it measured.

    Attributes:
        source_count: How many sources sounded together.
        phase_transform_exponent: The `beta` the curves were whitened with.
        pairwise_combinator: How pairwise maps were merged.
        pooling: Which interval pooling rule was applied.
        correlation: Correlation between the joint map and the summed
            individual maps, both at unit mass.
        normalized_residual: Norm of their difference over the norm of the
            joint map.
        grid_spacing_m: Cell size the maps were evaluated on.
    """

    source_count: int
    phase_transform_exponent: float
    pairwise_combinator: str
    pooling: str
    correlation: float
    normalized_residual: float
    grid_spacing_m: float

    def to_dict(self) -> dict[str, Any]:
        """Renders as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "source_count": self.source_count,
            "phase_transform_exponent": self.phase_transform_exponent,
            "pairwise_combinator": self.pairwise_combinator,
            "pooling": self.pooling,
            "correlation": self.correlation,
            "normalized_residual": self.normalized_residual,
            "grid_spacing_m": self.grid_spacing_m,
        }


def parse_arguments() -> argparse.Namespace:
    """Reads the configuration directory and the swept axes.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    parser.add_argument("--source-counts", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument(
        "--phase-transform-exponents",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.7, 1.0],
    )
    parser.add_argument(
        "--pairwise-combinators", type=str, nargs="+", default=["sum", "product"]
    )
    parser.add_argument("--poolings", type=str, nargs="+", default=["mean", "max"])
    parser.add_argument("--audit-grid-spacing-m", type=float, default=2.0)
    return parser.parse_args()


def build_source_positions_xyz_m(
    source_count: int,
    centre_xy_m: Float64Array,
    radius_m: float,
    source_height_m: float,
) -> Float64Array:
    """Places sources evenly on a ring, standing in for a burning contour.

    A ring rather than a scatter, because the distributed source this whole
    plan exists to image is a closed curve and linearity is worth measuring on
    the geometry it will actually be applied to.

    Args:
        source_count: How many sources to place.
        centre_xy_m: Ring centre, shape `(2,)`.
        radius_m: Ring radius, in metres.
        source_height_m: Height above the ground plane, in metres.

    Returns:
        Positions of shape `(source_count, 3)`.
    """
    angles_rad = np.linspace(0.0, 2.0 * np.pi, source_count, endpoint=False)
    centre = np.asarray(centre_xy_m, dtype=np.float64)
    return np.stack(
        (
            centre[0] + radius_m * np.cos(angles_rad),
            centre[1] + radius_m * np.sin(angles_rad),
            np.full(source_count, source_height_m, dtype=np.float64),
        ),
        axis=1,
    )


def render_sources(
    source_positions_xyz_m: Float64Array,
    source_indices: list[int],
    receiver_positions_xyz_m: Float64Array,
    configuration: SimulationConfiguration,
    sample_rate_hz: int,
) -> Float64Array:
    """Renders a chosen subset of the sources to the array, noiselessly.

    Noiselessly on purpose: receiver noise is common to the joint render and
    absent from the individual ones, so including it would show up as a
    linearity failure that has nothing to do with how the map is formed.

    Args:
        source_positions_xyz_m: Every source, shape `(n_sources, 3)`.
        source_indices: Which of them to sound.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        configuration: The scene, for air conditions and reference distance.
        sample_rate_hz: Sample rate in hertz.

    Returns:
        Receiver channels of shape `(n_receivers, n_samples)`.
    """
    signals = [
        generate_source_signal_for_burning_cell(
            1.0,
            AUDIT_SEGMENT_DURATION_S,
            sample_rate_hz,
            seed=SOURCE_SIGNAL_SEED + source_index,
        )
        for source_index in source_indices
    ]
    return render_multi_source_receiver_signals(
        signals,
        source_positions_xyz_m[source_indices],
        np.ones(len(source_indices), dtype=np.float64),
        receiver_positions_xyz_m,
        sample_rate_hz,
        configuration.atmosphere,
        configuration.forward_model.propagation.reference_distance_m,
    )


def compute_pairwise_pooled_maps(
    curves: list[GeneralizedCrossCorrelationCurve],
    delay_lower_bounds_s: Float64Array,
    delay_upper_bounds_s: Float64Array,
    pooling: str,
) -> Float64Array:
    """Pools every pair's curve over the grid, before any combination.

    Takes curves rather than waveforms so that one correlation pass serves
    every pooling rule, and one pooling pass serves every combinator. Whitening
    is the expensive step and it depends only on the exponent, so recomputing
    it per pooling rule would double the sweep for nothing.

    Args:
        curves: One correlation curve per pair.
        delay_lower_bounds_s: Lower cell bounds, shape `(n_cells, n_pairs)`.
        delay_upper_bounds_s: Upper cell bounds, same shape.
        pooling: Which interval pooling rule.

    Returns:
        Pooled maps of shape `(n_pairs, n_cells)`.
    """
    return np.stack(
        [
            pool_curve_over_intervals(
                curve,
                delay_lower_bounds_s[:, pair_index],
                delay_upper_bounds_s[:, pair_index],
                pooling,
            )
            for pair_index, curve in enumerate(curves)
        ]
    )


def compare_maps(
    joint_map: Float64Array, summed_map: Float64Array
) -> tuple[float, float]:
    """Scores a joint map against the sum of its parts, both at unit mass.

    Args:
        joint_map: Map of every source sounding together, shape `(n_cells,)`.
        summed_map: Sum of the individual maps, shape `(n_cells,)`.

    Returns:
        `(correlation, normalized_residual)`.
    """
    joint = normalize_map_to_unit_mass(np.clip(joint_map, 0.0, None))
    summed = normalize_map_to_unit_mass(np.clip(summed_map, 0.0, None))
    joint_norm = float(np.linalg.norm(joint))
    if joint_norm <= 0.0:
        return 0.0, float("inf")
    correlation = float(np.corrcoef(joint, summed)[0, 1])
    return correlation, float(np.linalg.norm(joint - summed) / joint_norm)


def measure_linearity(
    source_counts: list[int],
    phase_transform_exponents: list[float],
    pairwise_combinators: list[str],
    poolings: list[str],
    grid_spacing_m: float,
    configuration: SimulationConfiguration,
    receiver_positions_xyz_m: Float64Array,
    sample_rate_hz: int,
) -> list[LinearityCell]:
    """Sweeps every combination and measures how far each is from additive.

    Args:
        source_counts: Source counts to sweep.
        phase_transform_exponents: Exponents to sweep.
        pairwise_combinators: Combinators to sweep.
        poolings: Pooling rules to sweep.
        grid_spacing_m: Cell size the maps are evaluated on.
        configuration: The scene.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        sample_rate_hz: Sample rate in hertz.

    Returns:
        One cell per combination.
    """
    domain = configuration.geometry.domain
    multi_source = configuration.localization.multi_source
    receiver_pairs = enumerate_receiver_pairs(receiver_positions_xyz_m.shape[0])
    speed_of_sound_m_per_s = compute_speed_of_sound_m_per_s(
        configuration.atmosphere.air_temperature_celsius
    )
    maximum_absolute_lag_s = compute_maximum_absolute_lag_s(
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
        multi_source.correlation.maximum_absolute_lag_margin,
    )
    grid = build_candidate_grid_xyz(
        domain.size_x_m,
        domain.size_y_m,
        grid_spacing_m,
        multi_source.steered_response_power.candidate_height_m,
    )
    lower_s, upper_s = compute_cell_delay_bounds_s(
        grid.positions_xyz_m,
        grid.cell_extent_m,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
    )
    centre_xy_m = np.array([0.5 * domain.size_x_m, 0.5 * domain.size_y_m])

    cells: list[LinearityCell] = []
    for source_count in source_counts:
        positions_xyz_m = build_source_positions_xyz_m(
            source_count,
            centre_xy_m,
            SOURCE_RING_RADIUS_M,
            configuration.geometry.source_height_m,
        )
        renders = [
            render_sources(
                positions_xyz_m,
                list(range(source_count)),
                receiver_positions_xyz_m,
                configuration,
                sample_rate_hz,
            )
        ] + [
            render_sources(
                positions_xyz_m,
                [source_index],
                receiver_positions_xyz_m,
                configuration,
                sample_rate_hz,
            )
            for source_index in range(source_count)
        ]
        for exponent in phase_transform_exponents:
            correlation_configuration = CorrelationConfiguration(
                lowest_frequency_hz=multi_source.correlation.lowest_frequency_hz,
                highest_frequency_hz=multi_source.correlation.highest_frequency_hz,
                phase_transform_exponent=exponent,
                phase_transform_regularization=(
                    multi_source.correlation.phase_transform_regularization
                ),
                use_analytic_envelope=multi_source.correlation.use_analytic_envelope,
                window_duration_s=multi_source.correlation.window_duration_s,
                window_overlap=multi_source.correlation.window_overlap,
                maximum_absolute_lag_margin=(
                    multi_source.correlation.maximum_absolute_lag_margin
                ),
                peak_interpolation=multi_source.correlation.peak_interpolation,
            )
            curve_sets = [
                compute_pair_correlation_curves(
                    render,
                    receiver_pairs,
                    maximum_absolute_lag_s,
                    sample_rate_hz,
                    correlation_configuration,
                )
                for render in renders
            ]
            for pooling in poolings:
                pooled = [
                    compute_pairwise_pooled_maps(curves, lower_s, upper_s, pooling)
                    for curves in curve_sets
                ]
                for combinator in pairwise_combinators:
                    joint_map = combine_pairwise_maps(pooled[0], combinator)
                    summed_map = np.sum(
                        [
                            normalize_map_to_unit_mass(
                                np.clip(
                                    combine_pairwise_maps(one, combinator), 0.0, None
                                )
                            )
                            for one in pooled[1:]
                        ],
                        axis=0,
                    )
                    correlation, residual = compare_maps(joint_map, summed_map)
                    cells.append(
                        LinearityCell(
                            source_count=source_count,
                            phase_transform_exponent=exponent,
                            pairwise_combinator=combinator,
                            pooling=pooling,
                            correlation=correlation,
                            normalized_residual=residual,
                            grid_spacing_m=grid_spacing_m,
                        )
                    )
                print(
                    f"  K={source_count:2d} beta={exponent:.1f} pooling={pooling:5s} "
                    + "  ".join(
                        f"{cell.pairwise_combinator}: r={cell.correlation:.4f} "
                        f"res={cell.normalized_residual:.4f}"
                        for cell in cells[-len(pairwise_combinators) :]
                    )
                )
    return cells


def select_imaging_configuration(cells: list[LinearityCell]) -> LinearityCell:
    """Picks the most nearly additive combination at the gate's source count.

    Args:
        cells: Every measured combination.

    Returns:
        The cell with the smallest residual among those at the gate's source
        count, falling back to the largest count measured.

    Raises:
        ValueError: If nothing was measured.
    """
    if not cells:
        raise ValueError("the audit measured no combination")
    counts = {cell.source_count for cell in cells}
    target_count = GATE_SOURCE_COUNT if GATE_SOURCE_COUNT in counts else max(counts)
    candidates = [cell for cell in cells if cell.source_count == target_count]
    return min(candidates, key=lambda cell: cell.normalized_residual)


def main() -> None:
    """Runs the sweep, writes the report and the figure."""
    arguments = parse_arguments()
    configuration = load_simulation_configuration(arguments.configs)
    characterization = load_fire_characterization_configuration(
        arguments.configs / FIRE_CHARACTERIZATION_FILENAME
    )
    receiver_positions_xyz_m = place_receivers_from_layout(
        configuration.geometry.receiver_layout,
        configuration.geometry.domain.size_x_m,
        configuration.geometry.domain.size_y_m,
    )
    sample_rate_hz = 44100

    print(
        f"map linearity audit: {receiver_positions_xyz_m.shape[0]} receivers, "
        f"grid {arguments.audit_grid_spacing_m:.1f} m, "
        f"K in {arguments.source_counts}, beta in "
        f"{arguments.phase_transform_exponents}"
    )
    cells = measure_linearity(
        arguments.source_counts,
        arguments.phase_transform_exponents,
        arguments.pairwise_combinators,
        arguments.poolings,
        arguments.audit_grid_spacing_m,
        configuration,
        receiver_positions_xyz_m,
        sample_rate_hz,
    )
    chosen = select_imaging_configuration(cells)
    print(
        f"\nchosen imaging family: beta={chosen.phase_transform_exponent}, "
        f"combinator={chosen.pairwise_combinator}, pooling={chosen.pooling} "
        f"(r={chosen.correlation:.4f}, residual={chosen.normalized_residual:.4f})"
    )

    confirmation = measure_linearity(
        [chosen.source_count],
        [chosen.phase_transform_exponent],
        [chosen.pairwise_combinator],
        [chosen.pooling],
        characterization.imaging_map.grid_spacing_m,
        configuration,
        receiver_positions_xyz_m,
        sample_rate_hz,
    )[0]
    print(
        f"same family at the imaging spacing "
        f"{characterization.imaging_map.grid_spacing_m:.2f} m: "
        f"r={confirmation.correlation:.4f}, "
        f"residual={confirmation.normalized_residual:.4f}"
    )

    figure_directory = FIGURE_ROOT / arguments.configs.name
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure: Figure = plot_map_linearity_audit(
        [cell.source_count for cell in cells],
        [cell.phase_transform_exponent for cell in cells],
        [cell.pairwise_combinator for cell in cells],
        [cell.pooling for cell in cells],
        [cell.normalized_residual for cell in cells],
        [cell.correlation for cell in cells],
        LINEARITY_RESIDUAL_TARGET,
    )
    figure_path = figure_directory / "tier0_map_linearity.svg"
    figure.savefig(figure_path, bbox_inches="tight")

    metrics_path = METRICS_ROOT / "tier0_map_linearity.json"
    write_metrics_document(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration_directory": str(arguments.configs),
            "receiver_count": int(receiver_positions_xyz_m.shape[0]),
            "audit_grid_spacing_m": arguments.audit_grid_spacing_m,
            "imaging_grid_spacing_m": characterization.imaging_map.grid_spacing_m,
            "source_ring_radius_m": SOURCE_RING_RADIUS_M,
            "linearity_correlation_target": LINEARITY_CORRELATION_TARGET,
            "linearity_residual_target": LINEARITY_RESIDUAL_TARGET,
            "chosen_imaging_family": chosen.to_dict(),
            "chosen_family_at_imaging_spacing": confirmation.to_dict(),
            "cells": [cell.to_dict() for cell in cells],
        },
        metrics_path,
    )
    print(f"\nfigure  {figure_path}")
    print(f"metrics {metrics_path}")


if __name__ == "__main__":
    main()
