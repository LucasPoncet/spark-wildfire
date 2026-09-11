"""Recovers where a fire's front is, by three routes, and scores all three.

Usage:
    uv run python scripts/run_front_characterization.py --configs configs/f1

Tier 3. Each frame's imaging map is read three ways — a matched filter on one
radial profile, a parametric elliptical arc, and a free-form deconvolution —
and each produces a front distance per direction, scored against the cells that
were actually alight.

Running all three on the same frames is the point rather than a thoroughness
gesture: the plan's own gate asks whether the free-form route beats the
parametric one, and says to prefer the parametric fit and report it if not.

The angular sweep is the whole circle, and the fronts cover about a quarter of
it. Directions the front does not reach carry no distance rather than a zero,
so coverage is a measured output on both the estimated and the true side.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_fire_characterization import CharacterizationFrame, build_frames
from src.config.fire_characterization_configuration import (
    FIRE_CHARACTERIZATION_FILENAME,
    FireCharacterizationConfiguration,
    load_fire_characterization_configuration,
)
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    SimulationConfiguration,
    load_forward_simulation_configuration,
    load_simulation_configuration,
)
from src.spark.inverse.fire_bearing import estimate_bearing_from_centroid_offset
from src.spark.inverse.front_contour import (
    compute_angular_coverage_fraction,
    compute_contour_distance_by_angle,
    extract_front_contour,
)
from src.spark.inverse.front_perimeter_fit import (
    PerimeterParameters,
    compute_front_distance_by_angle,
    fit_front_perimeter,
)
from src.spark.inverse.front_radial_profile import (
    estimate_front_band_by_matched_filter,
    estimate_front_distance_by_matched_filter,
    extract_radial_profile,
)
from src.spark.inverse.map_deconvolution import (
    build_point_spread_operator,
    deconvolve_with_total_variation,
)
from src.spark.inverse.map_normalization import prepare_map_for_moments
from src.spark.inverse.point_spread_function import compute_point_spread_semi_axes_m
from src.spark.inverse.steered_response_power_sequence import compute_imaging_grid_shape
from src.utils.array_types import Float64Array
from src.utils.io.metrics_writer import write_metrics_document
from src.utils.metrics.contour_metrics import (
    compute_coverage_agreement,
    compute_hausdorff_distance,
    compute_mean_radial_error,
    compute_sector_overlap,
)
from src.utils.metrics.fire_front_ground_truth import compute_true_distance_by_angle_m
from src.utils.visualization.front_position_plotter import plot_front_position

FIGURE_ROOT: Path = Path("results/figures")
METRICS_ROOT: Path = Path("results/metrics")
ANGLE_COUNT: int = 180
ANGULAR_TOLERANCE_RAD: float = float(np.deg2rad(2.5))
PERIMETER_SAMPLE_COUNT: int = 240
PERIMETER_MAXIMUM_ITERATIONS: int = 400
HEAD_DISTANCE_TARGET_M: float = 1.5
RADIAL_ERROR_TARGET_M: float = 2.0
OVERLAP_TARGET: float = 0.6


@dataclass(frozen=True)
class FrontFrame:
    """One frame read three ways, beside the truth.

    Attributes:
        time_s: Simulated time.
        true_distances_m: True front radius per angle, `nan` where uncovered.
        profile_head_distance_m: Leading edge from the band fit, which is the
            quantity a head distance means.
        profile_lobe_distance_m: Where a single-lobe fit put the band's centre,
            kept because the gap between the two is the band's half-width.
        profile_is_one_sided: Whether that fit found no trailing lobe.
        profile_residual: Its fit residual.
        perimeter_distances_m: Radius per angle from the parametric arc.
        perimeter_correlation: How well that arc explained the map.
        deconvolved_distances_m: Radius per angle from the deconvolution.
        deconvolved_contour_xy_m: Those radii as points, shape `(n_covered, 2)`.
        response_semi_axis_m: The array response's own semi-axis.
        true_head_distance_m: True distance from ignition to the head.
    """

    time_s: float
    true_distances_m: Float64Array
    profile_head_distance_m: float
    profile_lobe_distance_m: float
    profile_is_one_sided: bool
    profile_residual: float
    perimeter_distances_m: Float64Array
    perimeter_correlation: float
    deconvolved_distances_m: Float64Array
    deconvolved_contour_xy_m: Float64Array
    response_semi_axis_m: float
    true_head_distance_m: float


def parse_arguments() -> argparse.Namespace:
    """Reads the configuration directory and the optional window cap.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    parser.add_argument("--max-observations", type=int, default=None)
    return parser.parse_args()


def read_one_frame(
    frame: CharacterizationFrame,
    candidate_positions_xyz_m: Float64Array,
    grid_shape: tuple[int, int],
    ignition_xy_m: Float64Array,
    bearing_rad: float,
    characterization: FireCharacterizationConfiguration,
    angles_rad: Float64Array,
) -> FrontFrame:
    """Runs all three front-position routes on one frame.

    Args:
        frame: The window, carrying its map and its response.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        grid_shape: `(n_y, n_x)` of the imaging grid.
        ignition_xy_m: Estimated ignition point, shape `(2,)`.
        bearing_rad: Direction of spread, from the bearing tier.
        characterization: Front and imaging settings.
        angles_rad: Directions the front is evaluated on.

    Returns:
        What the three routes found, beside the truth.
    """
    front = characterization.front
    percentile = characterization.imaging_map.background_percentile
    prepared = prepare_map_for_moments(frame.map_values, percentile)
    prepared_response = prepare_map_for_moments(frame.response_map, percentile)

    direction = np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
    radii_m, profile = extract_radial_profile(
        prepared,
        candidate_positions_xyz_m,
        ignition_xy_m,
        direction,
        front.maximum_radius_m,
        front.radial_profile_step_m,
    )
    response_centre_xy_m = candidate_positions_xyz_m[
        int(np.argmax(frame.response_map)), :2
    ]
    response_radii_m, response_profile = extract_radial_profile(
        prepared_response,
        candidate_positions_xyz_m,
        response_centre_xy_m,
        direction,
        front.maximum_radius_m,
        front.radial_profile_step_m,
    )
    response_peak = float(np.max(response_profile))
    if response_peak > 0.0:
        response_profile = response_profile / response_peak
    distance = estimate_front_distance_by_matched_filter(
        radii_m, profile, response_radii_m, response_profile, front.maximum_radius_m
    )
    band = estimate_front_band_by_matched_filter(
        radii_m, profile, response_radii_m, response_profile, front.maximum_radius_m
    )

    operator = build_point_spread_operator(
        frame.response_map, grid_shape, characterization.imaging_map.grid_spacing_m
    )
    seed = PerimeterParameters(
        centre_xy_m=np.asarray(frame.map_centroid_xy_m, dtype=np.float64),
        semi_axis_major_m=max(frame.extent.semi_axis_major_m, 1.0),
        semi_axis_minor_m=max(frame.extent.semi_axis_minor_m, 0.6),
        orientation_rad=frame.extent.orientation_rad,
        sector_centre_rad=bearing_rad,
        sector_half_width_rad=float(np.deg2rad(60.0)),
    )
    perimeter = fit_front_perimeter(
        prepared,
        candidate_positions_xyz_m,
        operator,
        ignition_xy_m,
        seed,
        PERIMETER_MAXIMUM_ITERATIONS,
        PERIMETER_SAMPLE_COUNT,
    )
    perimeter_distances_m = compute_front_distance_by_angle(
        perimeter.parameters, ignition_xy_m, angles_rad
    )

    density = deconvolve_with_total_variation(
        prepared,
        operator,
        front.total_variation_weight,
        front.deconvolution_iteration_count,
    )
    deconvolved_distances_m = compute_contour_distance_by_angle(
        density.ravel(),
        candidate_positions_xyz_m,
        ignition_xy_m,
        angles_rad,
        front.contour_level_fraction,
        front.maximum_radius_m,
        front.radial_profile_step_m,
    )
    deconvolved_contour_xy_m = extract_front_contour(
        density.ravel(),
        candidate_positions_xyz_m,
        ignition_xy_m,
        angles_rad,
        front.contour_level_fraction,
        front.maximum_radius_m,
        front.radial_profile_step_m,
    )
    true_distances_m = compute_true_distance_by_angle_m(
        frame.truth.active_cell_positions_xy_m,
        ignition_xy_m,
        angles_rad,
        ANGULAR_TOLERANCE_RAD,
    )
    return FrontFrame(
        time_s=frame.time_s,
        true_distances_m=true_distances_m,
        profile_head_distance_m=band.outer_edge_m,
        profile_lobe_distance_m=distance.head_distance_m,
        profile_is_one_sided=distance.is_one_sided,
        profile_residual=band.residual,
        perimeter_distances_m=perimeter_distances_m,
        perimeter_correlation=perimeter.correlation,
        deconvolved_distances_m=deconvolved_distances_m,
        deconvolved_contour_xy_m=deconvolved_contour_xy_m,
        response_semi_axis_m=float(
            compute_point_spread_semi_axes_m(frame.response_covariance_m2)[0]
        ),
        true_head_distance_m=float(frame.truth.head_distance_m),
    )


def score_frame(front_frame: FrontFrame, angles_rad: Float64Array) -> dict[str, Any]:
    """Scores one frame's three routes against the truth.

    Args:
        front_frame: What the three routes found.
        angles_rad: Directions they were evaluated on.

    Returns:
        A JSON-serialisable mapping.
    """
    del angles_rad
    true_distances_m = front_frame.true_distances_m
    return {
        "time_s": front_frame.time_s,
        "response_semi_axis_m": front_frame.response_semi_axis_m,
        "true_head_distance_m": front_frame.true_head_distance_m,
        "true_coverage_fraction": compute_angular_coverage_fraction(true_distances_m),
        "profile": {
            "head_distance_m": front_frame.profile_head_distance_m,
            "lobe_centre_distance_m": front_frame.profile_lobe_distance_m,
            "head_error_m": abs(
                front_frame.profile_head_distance_m - front_frame.true_head_distance_m
            ),
            "is_one_sided": front_frame.profile_is_one_sided,
            "residual": front_frame.profile_residual,
        },
        "perimeter": {
            "mean_radial_error_m": compute_mean_radial_error(
                front_frame.perimeter_distances_m, true_distances_m
            ),
            "sector_overlap": compute_sector_overlap(
                front_frame.perimeter_distances_m, true_distances_m
            ),
            "coverage_agreement": compute_coverage_agreement(
                front_frame.perimeter_distances_m, true_distances_m
            ),
            "correlation": front_frame.perimeter_correlation,
        },
        "deconvolution": {
            "mean_radial_error_m": compute_mean_radial_error(
                front_frame.deconvolved_distances_m, true_distances_m
            ),
            "sector_overlap": compute_sector_overlap(
                front_frame.deconvolved_distances_m, true_distances_m
            ),
            "coverage_agreement": compute_coverage_agreement(
                front_frame.deconvolved_distances_m, true_distances_m
            ),
        },
    }


def build_gate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluates the Gate 3 criteria that this scene can answer.

    Args:
        records: One scored record per frame, in time order.

    Returns:
        One mapping per criterion.
    """
    resolvable = [
        record
        for record in records
        if record["true_head_distance_m"] > record["response_semi_axis_m"]
    ]
    head_errors_m = [record["profile"]["head_error_m"] for record in resolvable]
    final = records[-1]
    perimeter_error_m = final["perimeter"]["mean_radial_error_m"]
    deconvolution_error_m = final["deconvolution"]["mean_radial_error_m"]
    best_overlap = max(
        final["perimeter"]["sector_overlap"],
        final["deconvolution"]["sector_overlap"],
    )
    return [
        {
            "name": "matched-filter head distance, frames past one response semi-axis",
            "target": f"within {HEAD_DISTANCE_TARGET_M} m",
            "measured": (
                f"worst {max(head_errors_m):.2f} m over {len(resolvable)} frames"
                if head_errors_m
                else "no frame qualified"
            ),
            "passed": bool(head_errors_m)
            and max(head_errors_m) <= HEAD_DISTANCE_TARGET_M,
        },
        {
            "name": "parametric perimeter mean radial error, final frame",
            "target": f"below {RADIAL_ERROR_TARGET_M} m",
            "measured": f"{perimeter_error_m:.2f} m",
            "passed": bool(np.isfinite(perimeter_error_m))
            and perimeter_error_m < RADIAL_ERROR_TARGET_M,
        },
        {
            "name": "free-form deconvolution no worse than the parametric fit",
            "target": "<= parametric",
            "measured": (
                f"{deconvolution_error_m:.2f} m against {perimeter_error_m:.2f} m"
            ),
            "passed": bool(np.isfinite(deconvolution_error_m))
            and bool(np.isfinite(perimeter_error_m))
            and deconvolution_error_m <= perimeter_error_m,
        },
        {
            "name": "sector overlap on the final frame, better of the two routes",
            "target": f"above {OVERLAP_TARGET}",
            "measured": f"{best_overlap:.2f}",
            "passed": best_overlap > OVERLAP_TARGET,
        },
        {
            "name": "resolution onset curve is monotone in fire size",
            "target": "monotone",
            "measured": _describe_onset(records),
            "passed": _is_onset_monotone(records),
        },
    ]


def main() -> None:
    """Runs all three routes on every window and writes the report."""
    arguments = parse_arguments()
    forward_config = load_forward_simulation_configuration(arguments.configs)
    configuration: SimulationConfiguration = load_simulation_configuration(
        arguments.configs
    )
    characterization = load_fire_characterization_configuration(
        arguments.configs / FIRE_CHARACTERIZATION_FILENAME
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
        f"{forward_config.experiment.name}: front position by three routes, "
        f"{observation_count} windows"
    )
    frames, _, candidates_xyz_m, _true_ignition_xy_m = build_frames(
        forward_config,
        configuration,
        characterization,
        characterization.imaging_map.background_percentile,
        characterization.imaging_map.extent_background_percentile,
        observation_count,
    )
    domain = configuration.geometry.domain
    grid_shape = compute_imaging_grid_shape(
        domain.size_x_m, domain.size_y_m, characterization.imaging_map.grid_spacing_m
    )
    estimated_ignition_xy_m = np.asarray(frames[0].map_centroid_xy_m, dtype=np.float64)
    bearing = estimate_bearing_from_centroid_offset(
        frames[-1].map_values,
        candidates_xyz_m,
        estimated_ignition_xy_m,
        characterization.imaging_map.background_percentile,
        characterization.bearing.minimum_centroid_displacement_m,
    )
    angles_rad = np.linspace(0.0, 2.0 * np.pi, ANGLE_COUNT, endpoint=False)

    print(
        f"  bearing {np.rad2deg(bearing.bearing_rad):.2f} deg, ignition estimated at "
        f"({estimated_ignition_xy_m[0]:.2f}, {estimated_ignition_xy_m[1]:.2f}) m"
    )
    print(
        f"{'t':>5} {'true head':>10} {'profile':>9} {'err':>6} {'one-sided':>10} "
        f"{'perim err':>10} {'decon err':>10} {'overlap':>8}"
    )
    front_frames: list[FrontFrame] = []
    records: list[dict[str, Any]] = []
    for frame in frames:
        front_frame = read_one_frame(
            frame,
            candidates_xyz_m,
            grid_shape,
            estimated_ignition_xy_m,
            bearing.bearing_rad,
            characterization,
            angles_rad,
        )
        front_frames.append(front_frame)
        record = score_frame(front_frame, angles_rad)
        records.append(record)
        print(
            f"{record['time_s']:5.0f} {record['true_head_distance_m']:10.2f} "
            f"{record['profile']['head_distance_m']:9.2f} "
            f"{record['profile']['head_error_m']:6.2f} "
            f"{record['profile']['is_one_sided']!s:>10} "
            f"{record['perimeter']['mean_radial_error_m']:10.2f} "
            f"{record['deconvolution']['mean_radial_error_m']:10.2f} "
            f"{record['deconvolution']['sector_overlap']:8.2f}"
        )

    gate = build_gate_records(records)
    print("\nGATE 3")
    for criterion in gate:
        mark = "PASS" if criterion["passed"] else "FAIL"
        print(f"  [{mark}] {criterion['name']}")
        print(
            f"         target {criterion['target']}, measured {criterion['measured']}"
        )
    print(f"\n{sum(c['passed'] for c in gate)} of {len(gate)} criteria hold")

    figure_directory = FIGURE_ROOT / arguments.configs.name
    figure_directory.mkdir(parents=True, exist_ok=True)
    final = front_frames[-1]
    figure = plot_front_position(
        angles_rad,
        final.true_distances_m,
        final.perimeter_distances_m,
        final.deconvolved_distances_m,
        np.array([record["time_s"] for record in records]),
        np.array([record["profile"]["head_distance_m"] for record in records]),
        np.array([record["true_head_distance_m"] for record in records]),
        np.array([record["response_semi_axis_m"] for record in records]),
    )
    figure_path = figure_directory / "tier3_front_position.svg"
    figure.savefig(figure_path, bbox_inches="tight")

    metrics_path = METRICS_ROOT / "tier3_front_position.json"
    write_metrics_document(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration_directory": str(arguments.configs),
            "angle_count": ANGLE_COUNT,
            "bearing_deg": float(np.rad2deg(bearing.bearing_rad)),
            "estimated_ignition_xy_m": estimated_ignition_xy_m.tolist(),
            "deconvolution": {
                "iteration_count": characterization.front.deconvolution_iteration_count,
                "total_variation_weight": characterization.front.total_variation_weight,
                "contour_level_fraction": characterization.front.contour_level_fraction,
            },
            "hausdorff_final_frame_m": compute_hausdorff_distance(
                final.deconvolved_contour_xy_m,
                frames[-1].truth.active_cell_positions_xy_m,
            ),
            "frames": records,
            "gate_criteria": gate,
            "gate_passed": all(criterion["passed"] for criterion in gate),
        },
        metrics_path,
    )
    print(f"\nfigure  {figure_path}")
    print(f"metrics {metrics_path}")


def _describe_onset(records: list[dict[str, Any]]) -> str:
    """Summarises how the deconvolution error moves with fire size.

    Args:
        records: One scored record per frame, in time order.

    Returns:
        A short description of the trend.
    """
    ratios, errors = _collect_onset(records)
    if ratios.size < 3:
        return "too few resolvable frames"
    return (
        f"{errors[0]:.2f} m at {ratios[0]:.1f} response widths to "
        f"{errors[-1]:.2f} m at {ratios[-1]:.1f}"
    )


def _is_onset_monotone(records: list[dict[str, Any]]) -> bool:
    """Whether the error falls as the fire grows relative to the response.

    Args:
        records: One scored record per frame, in time order.

    Returns:
        True when the error's trend against fire size is downward.
    """
    ratios, errors = _collect_onset(records)
    if ratios.size < 3:
        return False
    slope = float(np.polyfit(ratios, errors, 1)[0])
    return slope < 0.0


def _collect_onset(
    records: list[dict[str, Any]],
) -> tuple[Float64Array, Float64Array]:
    """Fire size in response widths against the deconvolution's radial error.

    Args:
        records: One scored record per frame, in time order.

    Returns:
        `(size_ratios, errors_m)`, over frames where both are finite.
    """
    ratios: list[float] = []
    errors: list[float] = []
    for record in records:
        error_m = record["deconvolution"]["mean_radial_error_m"]
        if not np.isfinite(error_m):
            continue
        ratios.append(
            record["true_head_distance_m"] / max(record["response_semi_axis_m"], 1e-9)
        )
        errors.append(float(error_m))
    return np.array(ratios, dtype=np.float64), np.array(errors, dtype=np.float64)


if __name__ == "__main__":
    main()
