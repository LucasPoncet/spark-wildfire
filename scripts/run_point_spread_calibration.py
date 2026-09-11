"""Tabulates the array response across the domain and validates it against renders.

Usage:
    uv run python scripts/run_point_spread_calibration.py --configs configs/f1

Produces two fields the characterisation tiers interpolate rather than
recompute: the response covariance to subtract from an observed second moment,
and the centroid offset to subtract from an observed centroid.

Both are built from the analytic response rather than from a rendered source at
every node, and that shortcut is the thing this script has to earn. It earns it
by rendering point sources at a handful of validation positions and reporting
how well the analytic response reproduces them — the same check GATE 0 asks
for. If that agreement fails, the fields are still written and the report says
they are not trustworthy; it does not quietly fall back to something else.

The saving is not marginal. A rendered node costs a twenty-channel propagation
plus a one-hundred-and-ninety-pair correlation before any map is formed; the
analytic node costs the map alone.
"""

import argparse
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
from src.config.multi_source_localization_configuration import (
    CorrelationConfiguration,
    SteeredResponsePowerConfiguration,
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
    compute_maximum_offset_magnitude_m,
    has_no_sign_flip_between_adjacent_nodes,
    save_centroid_bias_field,
)
from src.spark.inverse.map_moments import compute_map_moments
from src.spark.inverse.map_normalization import prepare_map_for_moments
from src.spark.inverse.point_spread_function import (
    compute_point_spread_function,
    compute_point_spread_semi_axes_m,
)
from src.spark.inverse.receiver_pair_index import (
    compute_maximum_absolute_lag_s,
    enumerate_receiver_pairs,
)
from src.spark.inverse.steered_response_power import (
    CandidateGrid,
    build_candidate_grid_xyz,
    compute_map_over_grid,
)
from src.spark.inverse.time_difference_of_arrival import compute_pair_correlation_curves
from src.utils.array_types import Float64Array, Int64Array
from src.utils.io.metrics_writer import write_metrics_document
from src.utils.visualization.point_spread_plotter import (
    plot_point_spread_calibration,
)

FIGURE_ROOT: Path = Path("results/figures")
METRICS_ROOT: Path = Path("results/metrics")
CALIBRATION_SEGMENT_DURATION_S: float = 4.0
SOURCE_SIGNAL_SEED: int = 8192
SAMPLE_RATE_HZ: int = 44100
VALIDATION_CORRELATION_TARGET: float = 0.95
VALIDATION_COVARIANCE_TOLERANCE: float = 0.10
CENTRE_BIAS_TARGET_M: float = 0.30


@dataclass(frozen=True)
class ResponseValidation:
    """One position where the analytic response was checked against a render.

    Attributes:
        position_xy_m: Where the point source sat, shape `(2,)`.
        correlation: Correlation between the analytic and rendered maps, both
            prepared identically.
        analytic_semi_axes_m: Major and minor semi-axes of the analytic
            response, shape `(2,)`.
        rendered_semi_axes_m: The same for the rendered response.
        semi_axis_relative_errors: Fractional difference on each axis.
    """

    position_xy_m: Float64Array
    correlation: float
    analytic_semi_axes_m: Float64Array
    rendered_semi_axes_m: Float64Array
    semi_axis_relative_errors: Float64Array

    def to_dict(self) -> dict[str, Any]:
        """Renders as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "position_xy_m": self.position_xy_m.tolist(),
            "correlation": self.correlation,
            "analytic_semi_axes_m": self.analytic_semi_axes_m.tolist(),
            "rendered_semi_axes_m": self.rendered_semi_axes_m.tolist(),
            "semi_axis_relative_errors": self.semi_axis_relative_errors.tolist(),
        }


def parse_arguments() -> argparse.Namespace:
    """Reads the configuration directory.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    parser.add_argument(
        "--calibration-grid-spacing-m",
        type=float,
        default=None,
        help="Overrides the spacing in fire_characterization.toml, for a "
        "cheaper field while developing.",
    )
    return parser.parse_args()


def build_imaging_settings(
    configuration: SimulationConfiguration,
    characterization: FireCharacterizationConfiguration,
) -> tuple[CorrelationConfiguration, SteeredResponsePowerConfiguration]:
    """Derives the imaging correlation and map settings from the scene's own.

    Args:
        configuration: The scene.
        characterization: The characterisation settings.

    Returns:
        `(correlation, steered_response_power)` for the imaging family.
    """
    multi_source = configuration.localization.multi_source
    return (
        characterization.imaging_map.apply_to_correlation(multi_source.correlation),
        characterization.imaging_map.apply_to_steered_response_power(
            multi_source.steered_response_power
        ),
    )


def build_calibration_nodes_xyz_m(
    domain_extent_x_m: float,
    domain_extent_y_m: float,
    spacing_m: float,
    height_m: float,
) -> tuple[Float64Array, tuple[int, int]]:
    """Lays down the nodes the two fields are tabulated at.

    Args:
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        spacing_m: Node spacing, in metres.
        height_m: Source height above the ground plane, in metres.

    Returns:
        `(nodes_xyz_m, (n_y, n_x))` with x varying fastest.
    """
    grid = build_candidate_grid_xyz(
        domain_extent_x_m, domain_extent_y_m, spacing_m, height_m
    )
    column_count = int(np.unique(grid.positions_xyz_m[:, 0]).size)
    row_count = int(np.unique(grid.positions_xyz_m[:, 1]).size)
    return grid.positions_xyz_m, (row_count, column_count)


def render_point_source_map(
    position_xyz_m: Float64Array,
    grid: CandidateGrid,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    maximum_absolute_lag_s: float,
    configuration: SimulationConfiguration,
    correlation: CorrelationConfiguration,
    steered_response_power: SteeredResponsePowerConfiguration,
) -> Float64Array:
    """Propagates one point source to the array and images it.

    Args:
        position_xyz_m: Where the source sits, shape `(3,)`.
        grid: The imaging grid.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        maximum_absolute_lag_s: Largest lag retained, in seconds.
        configuration: The scene, for air conditions and reference distance.
        correlation: Imaging correlation settings.
        steered_response_power: Imaging map settings.

    Returns:
        The rendered map, shape `(n_cells,)`.
    """
    receiver_signals = render_multi_source_receiver_signals(
        [
            generate_source_signal_for_burning_cell(
                1.0,
                CALIBRATION_SEGMENT_DURATION_S,
                SAMPLE_RATE_HZ,
                seed=SOURCE_SIGNAL_SEED,
            )
        ],
        np.asarray(position_xyz_m, dtype=np.float64).reshape(1, 3),
        np.ones(1, dtype=np.float64),
        receiver_positions_xyz_m,
        SAMPLE_RATE_HZ,
        configuration.atmosphere,
        configuration.forward_model.propagation.reference_distance_m,
    )
    return compute_map_over_grid(
        compute_pair_correlation_curves(
            receiver_signals,
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


def validate_against_render(
    validation_positions_xy_m: Float64Array,
    grid: CandidateGrid,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    maximum_absolute_lag_s: float,
    configuration: SimulationConfiguration,
    characterization: FireCharacterizationConfiguration,
    correlation: CorrelationConfiguration,
    steered_response_power: SteeredResponsePowerConfiguration,
) -> list[ResponseValidation]:
    """Compares the analytic response to a rendered one at chosen positions.

    Args:
        validation_positions_xy_m: Where to check, shape `(n_positions, 2)`.
        grid: The imaging grid.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        maximum_absolute_lag_s: Largest lag retained, in seconds.
        configuration: The scene.
        characterization: The characterisation settings.
        correlation: Imaging correlation settings.
        steered_response_power: Imaging map settings.

    Returns:
        One record per validation position.
    """
    background_percentile = characterization.imaging_map.background_percentile
    source_height_m = configuration.geometry.source_height_m
    records: list[ResponseValidation] = []
    for position_xy_m in np.atleast_2d(validation_positions_xy_m):
        position_xyz_m = np.array(
            [position_xy_m[0], position_xy_m[1], source_height_m], dtype=np.float64
        )
        analytic = compute_point_spread_function(
            position_xyz_m,
            grid,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            SAMPLE_RATE_HZ,
            maximum_absolute_lag_s,
            correlation,
            steered_response_power,
            None,
        )
        rendered = render_point_source_map(
            position_xyz_m,
            grid,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            maximum_absolute_lag_s,
            configuration,
            correlation,
            steered_response_power,
        )
        analytic_prepared = prepare_map_for_moments(analytic, background_percentile)
        rendered_prepared = prepare_map_for_moments(rendered, background_percentile)
        analytic_semi_axes_m = compute_point_spread_semi_axes_m(
            compute_map_moments(
                analytic, grid.positions_xyz_m, background_percentile
            ).covariance_xy_m2
        )
        rendered_semi_axes_m = compute_point_spread_semi_axes_m(
            compute_map_moments(
                rendered, grid.positions_xyz_m, background_percentile
            ).covariance_xy_m2
        )
        records.append(
            ResponseValidation(
                position_xy_m=np.asarray(position_xy_m, dtype=np.float64),
                correlation=float(
                    np.corrcoef(analytic_prepared, rendered_prepared)[0, 1]
                ),
                analytic_semi_axes_m=analytic_semi_axes_m,
                rendered_semi_axes_m=rendered_semi_axes_m,
                semi_axis_relative_errors=np.abs(
                    analytic_semi_axes_m - rendered_semi_axes_m
                )
                / np.maximum(rendered_semi_axes_m, 1e-9),
            )
        )
        print(
            f"  validation at ({position_xy_m[0]:5.1f}, {position_xy_m[1]:5.1f}): "
            f"r = {records[-1].correlation:.4f}, "
            f"analytic semi-axes {analytic_semi_axes_m[0]:.2f}/"
            f"{analytic_semi_axes_m[1]:.2f} m, rendered "
            f"{rendered_semi_axes_m[0]:.2f}/{rendered_semi_axes_m[1]:.2f} m"
        )
    return records


def build_response_fields(
    nodes_xyz_m: Float64Array,
    grid_shape: tuple[int, int],
    grid: CandidateGrid,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    maximum_absolute_lag_s: float,
    background_percentile: float,
    correlation: CorrelationConfiguration,
    steered_response_power: SteeredResponsePowerConfiguration,
) -> tuple[CentroidBiasField, Float64Array]:
    """Tabulates centroid offset and response covariance at every node.

    Args:
        nodes_xyz_m: Node positions, shape `(n_nodes, 3)`.
        grid_shape: `(n_y, n_x)` of the node grid.
        grid: The imaging grid.
        receiver_positions_xyz_m: The array, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        maximum_absolute_lag_s: Largest lag retained, in seconds.
        background_percentile: Percentile of a frame treated as background.
        correlation: Imaging correlation settings.
        steered_response_power: Imaging map settings.

    Returns:
        `(bias_field, covariances_m2)` with covariances of shape
        `(n_nodes, 2, 2)`.
    """
    node_count = nodes_xyz_m.shape[0]
    offsets_xy_m = np.empty((node_count, 2), dtype=np.float64)
    covariances_m2 = np.empty((node_count, 2, 2), dtype=np.float64)
    for node_index in range(node_count):
        response = compute_point_spread_function(
            nodes_xyz_m[node_index],
            grid,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            SAMPLE_RATE_HZ,
            maximum_absolute_lag_s,
            correlation,
            steered_response_power,
            None,
        )
        moments = compute_map_moments(
            response, grid.positions_xyz_m, background_percentile
        )
        offsets_xy_m[node_index] = moments.centroid_xy_m - nodes_xyz_m[node_index, :2]
        covariances_m2[node_index] = moments.covariance_xy_m2
        if node_index % 10 == 0:
            print(
                f"  node {node_index + 1}/{node_count} at "
                f"({nodes_xyz_m[node_index, 0]:5.1f}, "
                f"{nodes_xyz_m[node_index, 1]:5.1f}): offset "
                f"{float(np.linalg.norm(offsets_xy_m[node_index])):5.2f} m"
            )
    return (
        CentroidBiasField(
            node_positions_xy_m=np.asarray(nodes_xyz_m[:, :2], dtype=np.float64),
            offsets_xy_m=offsets_xy_m,
            grid_shape=grid_shape,
        ),
        covariances_m2,
    )


def save_covariance_field(
    path: Path, nodes_xyz_m: Float64Array, covariances_m2: Float64Array
) -> Path:
    """Writes the response covariance field to disk.

    Args:
        path: Destination file, created with its parents if absent.
        nodes_xyz_m: Node positions, shape `(n_nodes, 3)`.
        covariances_m2: One covariance per node, shape `(n_nodes, 2, 2)`.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        node_positions_xyz_m=np.asarray(nodes_xyz_m, dtype=np.float64),
        covariances_m2=np.asarray(covariances_m2, dtype=np.float64),
    )
    return path


def main() -> None:
    """Validates the analytic response, tabulates both fields, writes the report."""
    arguments = parse_arguments()
    configuration = load_simulation_configuration(arguments.configs)
    characterization = load_fire_characterization_configuration(
        arguments.configs / FIRE_CHARACTERIZATION_FILENAME
    )
    correlation, steered_response_power = build_imaging_settings(
        configuration, characterization
    )
    domain = configuration.geometry.domain
    receiver_positions_xyz_m = place_receivers_from_layout(
        configuration.geometry.receiver_layout, domain.size_x_m, domain.size_y_m
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
        characterization.imaging_map.grid_spacing_m,
        steered_response_power.candidate_height_m,
    )
    spacing_m = (
        arguments.calibration_grid_spacing_m
        if arguments.calibration_grid_spacing_m is not None
        else characterization.point_spread_function.calibration_grid_spacing_m
    )
    nodes_xyz_m, grid_shape = build_calibration_nodes_xyz_m(
        domain.size_x_m,
        domain.size_y_m,
        spacing_m,
        configuration.geometry.source_height_m,
    )

    centre_xy_m = np.array([0.5 * domain.size_x_m, 0.5 * domain.size_y_m])
    ring_radius_m = configuration.geometry.receiver_layout.ring_radius_m
    validation_positions_xy_m = np.array(
        [
            centre_xy_m,
            centre_xy_m + np.array([0.5 * ring_radius_m, 0.0]),
            centre_xy_m + np.array([0.85 * ring_radius_m, 0.0]),
        ]
    )

    print(
        f"point spread calibration: {receiver_positions_xyz_m.shape[0]} receivers, "
        f"imaging grid {characterization.imaging_map.grid_spacing_m:.2f} m "
        f"({grid.positions_xyz_m.shape[0]} cells), calibration nodes "
        f"{grid_shape[0]}x{grid_shape[1]} at {spacing_m:.1f} m"
    )
    validations = validate_against_render(
        validation_positions_xy_m,
        grid,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
        maximum_absolute_lag_s,
        configuration,
        characterization,
        correlation,
        steered_response_power,
    )
    bias_field, covariances_m2 = build_response_fields(
        nodes_xyz_m,
        grid_shape,
        grid,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
        maximum_absolute_lag_s,
        characterization.imaging_map.background_percentile,
        correlation,
        steered_response_power,
    )

    centre_node_index = int(
        np.argmin(np.linalg.norm(nodes_xyz_m[:, :2] - centre_xy_m, axis=1))
    )
    centre_bias_m = float(np.linalg.norm(bias_field.offsets_xy_m[centre_node_index]))
    bias_path = save_centroid_bias_field(
        characterization.point_spread_function.centroid_bias_field_path, bias_field
    )
    covariance_path = save_covariance_field(
        characterization.point_spread_function.covariance_field_path,
        nodes_xyz_m,
        covariances_m2,
    )

    figure_directory = FIGURE_ROOT / arguments.configs.name
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure = plot_point_spread_calibration(
        bias_field.node_positions_xy_m,
        bias_field.offsets_xy_m,
        np.array(
            [
                compute_point_spread_semi_axes_m(covariance_m2)[0]
                for covariance_m2 in covariances_m2
            ]
        ),
        grid_shape,
        receiver_positions_xyz_m,
        np.array([record.correlation for record in validations]),
        np.stack([record.position_xy_m for record in validations]),
    )
    figure_path = figure_directory / "tier0_point_spread_calibration.svg"
    figure.savefig(figure_path, bbox_inches="tight")

    metrics_path = METRICS_ROOT / "tier0_point_spread_calibration.json"
    write_metrics_document(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration_directory": str(arguments.configs),
            "receiver_count": int(receiver_positions_xyz_m.shape[0]),
            "imaging_grid_spacing_m": characterization.imaging_map.grid_spacing_m,
            "background_percentile": (
                characterization.imaging_map.background_percentile
            ),
            "calibration_grid_spacing_m": spacing_m,
            "calibration_node_count": int(nodes_xyz_m.shape[0]),
            "validation_correlation_target": VALIDATION_CORRELATION_TARGET,
            "validation_covariance_tolerance": VALIDATION_COVARIANCE_TOLERANCE,
            "validations": [record.to_dict() for record in validations],
            "minimum_validation_correlation": float(
                min(record.correlation for record in validations)
            ),
            "maximum_semi_axis_relative_error": float(
                max(
                    float(np.max(record.semi_axis_relative_errors))
                    for record in validations
                )
            ),
            "centre_bias_target_m": CENTRE_BIAS_TARGET_M,
            "centre_bias_m": centre_bias_m,
            "maximum_bias_m": compute_maximum_offset_magnitude_m(bias_field),
            "bias_field_has_no_sign_flip": has_no_sign_flip_between_adjacent_nodes(
                bias_field
            ),
            "centroid_bias_field_path": str(bias_path),
            "covariance_field_path": str(covariance_path),
        },
        metrics_path,
    )
    print(
        f"\ncentre bias {centre_bias_m:.3f} m, maximum bias "
        f"{compute_maximum_offset_magnitude_m(bias_field):.3f} m"
    )
    print(f"bias field       {bias_path}")
    print(f"covariance field {covariance_path}")
    print(f"figure           {figure_path}")
    print(f"metrics          {metrics_path}")


if __name__ == "__main__":
    main()
