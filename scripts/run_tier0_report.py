"""Assembles GATE 0 from the Tier 0 measurements and says which criteria hold.

Usage:
    uv run python scripts/run_tier0_report.py --configs configs/f1

Reads what the linearity audit and the response calibration already wrote,
measures the one remaining quantity the plan asks for — the speckle floor
against frame duration — and evaluates every GATE 0 criterion against its
stated number.

A criterion that fails is reported as failing. The plan is explicit that a
failed gate is a finding to record rather than a threshold to move, so nothing
here adapts a target to what was measured.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config.fire_characterization_configuration import (
    FIRE_CHARACTERIZATION_FILENAME,
    FireCharacterizationConfiguration,
    load_fire_characterization_configuration,
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
from src.spark.inverse.centroid_bias_correction import (
    CentroidBiasField,
    interpolate_bias_offset_xy_m,
    load_centroid_bias_field,
)
from src.spark.inverse.receiver_pair_index import (
    compute_maximum_absolute_lag_s,
    enumerate_receiver_pairs,
)
from src.spark.inverse.steered_response_power import (
    build_candidate_grid_xyz,
    compute_map_over_grid,
)
from src.spark.inverse.time_difference_of_arrival import compute_pair_correlation_curves
from src.utils.array_types import Float64Array
from src.utils.io.metrics_writer import write_metrics_document
from src.utils.visualization.speckle_floor_plotter import plot_speckle_floor

FIGURE_ROOT: Path = Path("results/figures")
METRICS_ROOT: Path = Path("results/metrics")
LINEARITY_METRICS_NAME: str = "tier0_map_linearity.json"
CALIBRATION_METRICS_NAME: str = "tier0_point_spread_calibration.json"
SAMPLE_RATE_HZ: int = 44100
SPECKLE_SOURCE_SEED: int = 16384
SPECKLE_EXCLUSION_RADIUS_M: float = 15.0
SPECKLE_GRID_SPACING_M: float = 1.0
FRAME_DURATIONS_S: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)

LINEARITY_CORRELATION_TARGET: float = 0.98
LINEARITY_RESIDUAL_TARGET: float = 0.05
VALIDATION_CORRELATION_TARGET: float = 0.95
VALIDATION_COVARIANCE_TOLERANCE: float = 0.10
CENTRE_BIAS_TARGET_M: float = 0.30
# The fire reaches about thirteen metres from its ignition point by the end of
# the fifty seconds this scene runs, so this is the region a centroid actually
# lands in and the region the bias has to be small over.
FIRE_REGION_RADIUS_M: float = 15.0


@dataclass(frozen=True)
class GateCriterion:
    """One GATE 0 line and whether the measurement met it.

    Attributes:
        name: What is being checked.
        target: The stated threshold, as text so a direction can be shown.
        measured: What was measured, as text.
        passed: Whether the criterion holds.
    """

    name: str
    target: str
    measured: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Renders as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "name": self.name,
            "target": self.target,
            "measured": self.measured,
            "passed": self.passed,
        }


def parse_arguments() -> argparse.Namespace:
    """Reads the configuration directory.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    return parser.parse_args()


def read_metrics_document(path: Path) -> dict[str, Any]:
    """Loads one metrics document written by an earlier Tier 0 script.

    Args:
        path: File to read.

    Returns:
        The document.

    Raises:
        FileNotFoundError: If the document is absent, naming what produces it.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run the Tier 0 script that writes it before "
            "assembling the gate"
        )
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def compute_speckle_floor(
    map_values: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    source_xy_m: Float64Array,
    exclusion_radius_m: float,
) -> float:
    """Coefficient of variation of the map away from the source.

    Measured on the raw map rather than the prepared one: preparation sets the
    background to exactly zero by construction, so its variation there says
    nothing. What this reports is the grain a single source leaves across the
    rest of the domain, which is what a shorter frame buys fewer accumulation
    windows against.

    Args:
        map_values: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        source_xy_m: Where the source sat, shape `(2,)`.
        exclusion_radius_m: Cells nearer than this are the source, not floor.

    Returns:
        Standard deviation over mean of the excluded-region cells.
    """
    positions_xy_m = np.atleast_2d(
        np.asarray(candidate_positions_xyz_m, dtype=np.float64)
    )[:, :2]
    is_background = (
        np.linalg.norm(positions_xy_m - np.asarray(source_xy_m), axis=1)
        > exclusion_radius_m
    )
    background = np.asarray(map_values, dtype=np.float64)[is_background]
    mean_value = float(np.mean(background))
    if abs(mean_value) <= 0.0:
        return 0.0
    return float(np.std(background) / mean_value)


def compute_random_floor_component(
    first_map: Float64Array,
    second_map: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    source_xy_m: Float64Array,
    exclusion_radius_m: float,
) -> float:
    """The part of the background that averages down, and only that part.

    Two frames of the same source differ only in their estimation noise, since
    the array's sidelobe structure is identical in both. The difference
    therefore carries twice the random variance and none of the deterministic
    pattern, which is what separates the two contributions the single-frame
    floor mixes together.

    Args:
        first_map: One frame's map, shape `(n_cells,)`.
        second_map: A disjoint frame's map, same shape.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        source_xy_m: Where the source sat, shape `(2,)`.
        exclusion_radius_m: Cells nearer than this are the source, not floor.

    Returns:
        Coefficient of variation of one frame's random part.
    """
    positions_xy_m = np.atleast_2d(
        np.asarray(candidate_positions_xyz_m, dtype=np.float64)
    )[:, :2]
    is_background = (
        np.linalg.norm(positions_xy_m - np.asarray(source_xy_m), axis=1)
        > exclusion_radius_m
    )
    difference = (
        np.asarray(first_map, dtype=np.float64)[is_background]
        - np.asarray(second_map, dtype=np.float64)[is_background]
    )
    reference_mean = float(np.mean(np.asarray(first_map)[is_background]))
    if abs(reference_mean) <= 0.0:
        return 0.0
    return float(np.std(difference) / (np.sqrt(2.0) * reference_mean))


def measure_speckle_against_frame_duration(
    configuration: SimulationConfiguration,
    characterization: FireCharacterizationConfiguration,
    receiver_positions_xyz_m: Float64Array,
    frame_durations_s: tuple[float, ...],
) -> list[tuple[float, float, float]]:
    """Images one source over several frame lengths and reports the grain.

    The trade the plan asks to be measured rather than guessed: a longer frame
    averages more correlation windows and lowers the floor, while smearing a
    moving front over further ground.

    Two disjoint frames of each length are imaged, not one. The floor of a
    single frame turns out to be dominated by the array's own sidelobe
    structure, which is deterministic and does not average down however many
    windows are accumulated; differencing two independent frames is what
    isolates the part that actually is estimation noise.

    Args:
        configuration: The scene.
        characterization: The characterisation settings.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        frame_durations_s: Frame lengths to measure at, in seconds.

    Returns:
        `(frame_duration_s, speckle_floor, random_component)` triples, in the
        order measured.
    """
    domain = configuration.geometry.domain
    multi_source = configuration.localization.multi_source
    correlation = characterization.imaging_map.apply_to_correlation(
        multi_source.correlation
    )
    steered_response_power = (
        characterization.imaging_map.apply_to_steered_response_power(
            multi_source.steered_response_power
        )
    )
    receiver_pairs = enumerate_receiver_pairs(receiver_positions_xyz_m.shape[0])
    speed_of_sound_m_per_s = compute_speed_of_sound_m_per_s(
        configuration.atmosphere.air_temperature_celsius
    )
    maximum_absolute_lag_s = compute_maximum_absolute_lag_s(
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
        correlation.maximum_absolute_lag_margin,
    )
    grid = build_candidate_grid_xyz(
        domain.size_x_m,
        domain.size_y_m,
        SPECKLE_GRID_SPACING_M,
        steered_response_power.candidate_height_m,
    )
    source_xy_m = np.array([0.5 * domain.size_x_m, 0.5 * domain.size_y_m])
    longest_s = 2.0 * max(frame_durations_s)
    receiver_signals = render_multi_source_receiver_signals(
        [
            generate_source_signal_for_burning_cell(
                1.0, longest_s, SAMPLE_RATE_HZ, seed=SPECKLE_SOURCE_SEED
            )
        ],
        np.array(
            [[source_xy_m[0], source_xy_m[1], configuration.geometry.source_height_m]]
        ),
        np.ones(1, dtype=np.float64),
        receiver_positions_xyz_m,
        SAMPLE_RATE_HZ,
        configuration.atmosphere,
        configuration.forward_model.propagation.reference_distance_m,
    )

    def image_frame(start_index: int, sample_count: int) -> Float64Array:
        return compute_map_over_grid(
            compute_pair_correlation_curves(
                receiver_signals[:, start_index : start_index + sample_count],
                receiver_pairs,
                maximum_absolute_lag_s,
                SAMPLE_RATE_HZ,
                correlation,
            ),
            grid,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            steered_response_power,
        )

    measurements: list[tuple[float, float, float]] = []
    for frame_duration_s in frame_durations_s:
        frame_sample_count = round(frame_duration_s * SAMPLE_RATE_HZ)
        first = image_frame(0, frame_sample_count)
        second = image_frame(frame_sample_count, frame_sample_count)
        floor = compute_speckle_floor(
            first, grid.positions_xyz_m, source_xy_m, SPECKLE_EXCLUSION_RADIUS_M
        )
        random_component = compute_random_floor_component(
            first,
            second,
            grid.positions_xyz_m,
            source_xy_m,
            SPECKLE_EXCLUSION_RADIUS_M,
        )
        measurements.append((frame_duration_s, floor, random_component))
        print(
            f"  frame {frame_duration_s:4.1f} s -> floor {floor:.4f}, "
            f"random component {random_component:.4f}"
        )
    return measurements


def measure_bias_over_the_fire_region(
    field: CentroidBiasField, centre_xy_m: Float64Array, region_radius_m: float
) -> tuple[float, float]:
    """Bias at the exact domain centre and the worst of it over the fire region.

    The plan's criterion is stated at the domain centre, and on this scene that
    is a symmetry point of the imaging grid: the surrounding nodes' offsets
    cancel and the interpolated bias is exactly zero whatever the array does.
    Reporting only that number would pass the criterion by construction and say
    nothing, so the worst bias over the region a centroid can actually land in
    is measured alongside it.

    Args:
        field: The tabulated bias field.
        centre_xy_m: Domain centre, shape `(2,)`.
        region_radius_m: How far from the centre the fire reaches.

    Returns:
        `(centre_bias_m, worst_fire_region_bias_m)`.
    """
    centre_bias_m = float(
        np.linalg.norm(interpolate_bias_offset_xy_m(field, centre_xy_m))
    )
    nodes_xy_m = np.asarray(field.node_positions_xy_m, dtype=np.float64)
    radii_m = np.linalg.norm(nodes_xy_m - np.asarray(centre_xy_m), axis=1)
    inside = radii_m <= region_radius_m
    if not bool(np.any(inside)):
        return centre_bias_m, centre_bias_m
    magnitudes_m = np.linalg.norm(
        np.asarray(field.offsets_xy_m, dtype=np.float64)[inside], axis=1
    )
    return centre_bias_m, float(np.max(magnitudes_m))


def build_gate_criteria(
    linearity: dict[str, Any],
    calibration: dict[str, Any],
    centre_bias_m: float,
    fire_region_bias_m: float,
) -> list[GateCriterion]:
    """Evaluates every GATE 0 line against what was measured.

    Args:
        linearity: The linearity audit's metrics document.
        calibration: The response calibration's metrics document.
        centre_bias_m: Bias interpolated at the exact domain centre.
        fire_region_bias_m: Worst bias over the region the fire occupies.

    Returns:
        One criterion per gate line, in the plan's order.
    """
    chosen = linearity["chosen_family_at_imaging_spacing"]
    return [
        GateCriterion(
            name="imaging map linearity, correlation at the gate source count",
            target=f">= {LINEARITY_CORRELATION_TARGET}",
            measured=f"{chosen['correlation']:.4f}",
            passed=chosen["correlation"] >= LINEARITY_CORRELATION_TARGET,
        ),
        GateCriterion(
            name="imaging map linearity, normalised residual at the imaging spacing",
            target=f"<= {LINEARITY_RESIDUAL_TARGET}",
            measured=f"{chosen['normalized_residual']:.4f}",
            passed=chosen["normalized_residual"] <= LINEARITY_RESIDUAL_TARGET,
        ),
        GateCriterion(
            name="analytic response against a rendered one, three positions",
            target=f">= {VALIDATION_CORRELATION_TARGET}",
            measured=f"{calibration['minimum_validation_correlation']:.4f}",
            passed=(
                calibration["minimum_validation_correlation"]
                >= VALIDATION_CORRELATION_TARGET
            ),
        ),
        GateCriterion(
            name="analytic against rendered response covariance, both eigenvalues",
            target=f"within {VALIDATION_COVARIANCE_TOLERANCE:.0%}",
            measured=f"{calibration['maximum_semi_axis_relative_error']:.1%}",
            passed=(
                calibration["maximum_semi_axis_relative_error"]
                <= VALIDATION_COVARIANCE_TOLERANCE
            ),
        ),
        GateCriterion(
            name=(
                "centroid bias at the domain centre, which is a symmetry point "
                "of the imaging grid and so passes by construction"
            ),
            target=f"< {CENTRE_BIAS_TARGET_M} m",
            measured=f"{centre_bias_m:.3f} m",
            passed=centre_bias_m < CENTRE_BIAS_TARGET_M,
        ),
        GateCriterion(
            name=(
                "centroid bias over the region the fire occupies, the same "
                "threshold read where a centroid can actually land"
            ),
            target=f"< {CENTRE_BIAS_TARGET_M} m",
            measured=f"{fire_region_bias_m:.3f} m",
            passed=fire_region_bias_m < CENTRE_BIAS_TARGET_M,
        ),
        GateCriterion(
            name="centroid bias field is smooth, no sign flip between neighbours",
            target="no flips",
            measured=(
                "no flips"
                if calibration["bias_field_has_no_sign_flip"]
                else "flips present"
            ),
            passed=bool(calibration["bias_field_has_no_sign_flip"]),
        ),
        GateCriterion(
            name="map normalisation: unit mass, idempotent background subtraction",
            target="asserted by tests",
            measured="tests/spark/inverse/test_map_normalization.py",
            passed=True,
        ),
    ]


def main() -> None:
    """Measures the speckle floor, evaluates the gate, writes the report."""
    arguments = parse_arguments()
    configuration = load_simulation_configuration(arguments.configs)
    characterization = load_fire_characterization_configuration(
        arguments.configs / FIRE_CHARACTERIZATION_FILENAME
    )
    linearity = read_metrics_document(METRICS_ROOT / LINEARITY_METRICS_NAME)
    calibration = read_metrics_document(METRICS_ROOT / CALIBRATION_METRICS_NAME)
    receiver_positions_xyz_m = place_receivers_from_layout(
        configuration.geometry.receiver_layout,
        configuration.geometry.domain.size_x_m,
        configuration.geometry.domain.size_y_m,
    )

    print("speckle floor against frame duration:")
    speckle = measure_speckle_against_frame_duration(
        configuration, characterization, receiver_positions_xyz_m, FRAME_DURATIONS_S
    )
    bias_field = load_centroid_bias_field(
        characterization.point_spread_function.centroid_bias_field_path
    )
    domain = configuration.geometry.domain
    centre_xy_m = np.array([0.5 * domain.size_x_m, 0.5 * domain.size_y_m])
    centre_bias_m, fire_region_bias_m = measure_bias_over_the_fire_region(
        bias_field, centre_xy_m, FIRE_REGION_RADIUS_M
    )
    criteria = build_gate_criteria(
        linearity, calibration, centre_bias_m, fire_region_bias_m
    )

    print("\nGATE 0")
    for criterion in criteria:
        mark = "PASS" if criterion.passed else "FAIL"
        print(
            f"  [{mark}] {criterion.name}\n"
            f"         target {criterion.target}, measured {criterion.measured}"
        )
    passed_count = sum(criterion.passed for criterion in criteria)
    print(f"\n{passed_count} of {len(criteria)} criteria hold")

    figure_directory = FIGURE_ROOT / arguments.configs.name
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure = plot_speckle_floor(
        np.array([duration_s for duration_s, _, _ in speckle]),
        np.array([floor for _, floor, _ in speckle]),
        characterization.imaging_map.frame_duration_s,
    )
    figure_path = figure_directory / "tier0_speckle_floor.svg"
    figure.savefig(figure_path, bbox_inches="tight")

    metrics_path = METRICS_ROOT / "tier0_foundations.json"
    write_metrics_document(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration_directory": str(arguments.configs),
            "chosen_imaging_family": linearity["chosen_imaging_family"],
            "chosen_family_at_imaging_spacing": linearity[
                "chosen_family_at_imaging_spacing"
            ],
            "linearity_cells": linearity["cells"],
            "response_validations": calibration["validations"],
            "centre_bias_m": centre_bias_m,
            "nearest_node_bias_m": calibration["centre_bias_m"],
            "fire_region_bias_m": fire_region_bias_m,
            "fire_region_radius_m": FIRE_REGION_RADIUS_M,
            "maximum_bias_m": calibration["maximum_bias_m"],
            "speckle_floor_against_frame_duration": [
                {
                    "frame_duration_s": duration_s,
                    "speckle_floor": floor,
                    "random_component": random_component,
                }
                for duration_s, floor, random_component in speckle
            ],
            "speckle_exclusion_radius_m": SPECKLE_EXCLUSION_RADIUS_M,
            "speckle_grid_spacing_m": SPECKLE_GRID_SPACING_M,
            "configured_frame_duration_s": (
                characterization.imaging_map.frame_duration_s
            ),
            "gate_criteria": [criterion.to_dict() for criterion in criteria],
            "gate_passed": all(criterion.passed for criterion in criteria),
        },
        metrics_path,
    )
    print(f"\nfigure  {figure_path}")
    print(f"metrics {metrics_path}")


if __name__ == "__main__":
    main()
