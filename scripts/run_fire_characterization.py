"""Estimates a fire's bearing and rate of spread from twenty receivers.

Usage:
    uv run python scripts/run_fire_characterization.py --configs configs/f1

Tiers 1 and 2 of `Related_works/FIRE_CHARACTERIZATION_PLAN.md`. The fire spreads
under the propagation-equation engine; every window is rendered to the array and
imaged with the imaging map family; and every quantity below is read off those
maps rather than off any detection.

That last point is the whole reframing the plan rests on. A burning front is a
distributed source and the deflation loop fits point sources to it, so the
detections are a lossy projection of the map. Bearing and rate of spread are
therefore moments of the map, which survive a front the array cannot resolve.

The array response is evaluated at each frame's own centroid rather than
interpolated from the calibration field. Ten frames cost ten responses, which is
cheaper than the field and exact where the field would interpolate.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_acoustic_rendering import extract_sources
from scripts.run_fire_shape_localization import (
    advance_fire_to_time_s,
    render_front_to_receivers,
    resolve_time_step_s,
)
from src.config.fire_characterization_configuration import (
    FIRE_CHARACTERIZATION_FILENAME,
    FireCharacterizationConfiguration,
    load_fire_characterization_configuration,
)
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    ForwardSimulationConfiguration,
    SimulationConfiguration,
    load_forward_simulation_configuration,
    load_simulation_configuration,
)
from src.config.simulation_context_factory import build_simulation_context
from src.spark.acoustic.receiver_placement import place_receivers_from_layout
from src.spark.atmosphere.atmospheric_conditions import compute_speed_of_sound_m_per_s
from src.spark.inverse.fire_bearing import (
    BearingEstimate,
    estimate_bearing_from_centroid_drift,
    estimate_bearing_from_centroid_offset,
    estimate_bearing_from_principal_axis,
    estimate_bearing_from_skewness,
    estimate_ignition_position_xy_m,
    fuse_bearing_estimates,
)
from src.spark.inverse.fire_extent import ExtentEstimate, estimate_fire_extent
from src.spark.inverse.fire_rate_of_spread import (
    RateOfSpreadEstimate,
    estimate_rate_of_spread,
)
from src.spark.inverse.map_moments import compute_map_moments
from src.spark.inverse.point_spread_function import (
    compute_point_spread_function,
    compute_point_spread_semi_axes_m,
)
from src.spark.inverse.receiver_pair_index import (
    compute_maximum_absolute_lag_s,
    enumerate_receiver_pairs,
)
from src.spark.inverse.steered_response_power import (
    build_candidate_grid_xyz,
    compute_map_over_grid,
)
from src.spark.inverse.steered_response_power_sequence import (
    SteeredResponsePowerSequence,
    compute_imaging_grid_shape,
)
from src.spark.inverse.time_difference_of_arrival import compute_pair_correlation_curves
from src.utils.array_types import Float64Array
from src.utils.io.metrics_writer import write_metrics_document
from src.utils.metrics.bearing_metrics import (
    compute_agreement_fraction,
    compute_bearing_error_series,
    compute_circular_error_rad,
    summarize_bearing_errors,
)
from src.utils.metrics.fire_front_ground_truth import (
    FireFrontTruth,
    extract_fire_front_truth,
)
from src.utils.metrics.spread_metrics import (
    compute_extent_error_series,
    compute_rate_of_spread_error,
    compute_true_rate_of_spread_m_per_s,
    is_flag_honest,
)
from src.utils.visualization.fire_characterization_plotter import (
    plot_fire_characterization,
)

FIGURE_ROOT: Path = Path("results/figures")
METRICS_ROOT: Path = Path("results/metrics")
SAMPLE_RATE_HZ: int = 44100
AGREEMENT_TOLERANCE_DEG: float = 15.0
SETTLED_TIME_S: float = 20.0


@dataclass(frozen=True)
class CharacterizationFrame:
    """One window: the map it produced and the truth it came from.

    Attributes:
        time_s: Simulated time at the end of the window.
        map_values: The imaging map, shape `(n_cells,)`.
        truth: What the fire actually did.
        response_covariance_m2: Array response covariance at this centroid.
        extent: What the extent stage made of the frame.
        skewness_bearing: Bearing from this frame's skewness alone.
        principal_axis_bearing: Bearing from its principal axis, when defined.
        centroid_offset_bearing: Bearing from ignition to this frame's centroid,
            when the centroid has left the ignition point at all.
        map_centroid_xy_m: The frame's centroid, shape `(2,)`.
        response_map: The array response at that centroid, shape `(n_cells,)`.
            Kept whole rather than reduced to its covariance because the front
            position tier deconvolves with it.
    """

    time_s: float
    map_values: Float64Array
    truth: FireFrontTruth
    response_covariance_m2: Float64Array
    extent: ExtentEstimate
    skewness_bearing: BearingEstimate
    principal_axis_bearing: BearingEstimate | None
    centroid_offset_bearing: BearingEstimate | None
    map_centroid_xy_m: Float64Array
    response_map: Float64Array


def parse_arguments() -> argparse.Namespace:
    """Reads the configuration directory and the optional window cap.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    parser.add_argument("--max-observations", type=int, default=None)
    parser.add_argument(
        "--background-percentile",
        type=float,
        default=None,
        help="Overrides the imaging percentile, for sweeping it without "
        "editing the scene.",
    )
    return parser.parse_args()


def build_frames(
    forward_config: ForwardSimulationConfiguration,
    configuration: SimulationConfiguration,
    characterization: FireCharacterizationConfiguration,
    background_percentile: float,
    extent_background_percentile: float,
    observation_count: int,
) -> tuple[list[CharacterizationFrame], Float64Array, Float64Array, Float64Array]:
    """Runs the fire, images every window and reduces each to its moments.

    Args:
        forward_config: Forward-side configuration.
        configuration: Localization-side configuration.
        characterization: Imaging and characterisation settings.
        background_percentile: Percentile used for first moments.
        extent_background_percentile: Percentile used for second moments.
        observation_count: How many windows to run.

    Returns:
        `(frames, receiver_positions_xyz_m, candidate_positions_xyz_m,
        ignition_position_xy_m)`.
    """
    context = build_simulation_context(forward_config)
    time_step_s = resolve_time_step_s(forward_config, context)
    domain = configuration.geometry.domain
    receiver_positions_xyz_m = place_receivers_from_layout(
        configuration.geometry.receiver_layout, domain.size_x_m, domain.size_y_m
    )
    receiver_pairs = enumerate_receiver_pairs(receiver_positions_xyz_m.shape[0])
    speed_of_sound_m_per_s = compute_speed_of_sound_m_per_s(
        forward_config.atmosphere.air_temperature_celsius
    )
    multi_source = configuration.localization.multi_source
    correlation = characterization.imaging_map.apply_to_correlation(
        multi_source.correlation
    )
    steered_response_power = (
        characterization.imaging_map.apply_to_steered_response_power(
            multi_source.steered_response_power
        )
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

    fire_state = context.spread_engine.initialize(
        context.mesh, context.fuel_field, context.wind_field
    )
    fire_state = context.spread_engine.ignite_cells(
        fire_state, context.ignition_cell_indices
    )
    ignition_position_xy_m = np.mean(
        context.mesh.cell_positions_xyz[context.ignition_cell_indices, :2], axis=0
    )
    cell_area_m2 = forward_config.mesh.cell_spacing_m**2

    frames: list[CharacterizationFrame] = []
    for observation_index in range(observation_count):
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
        map_values = compute_map_over_grid(
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
        moments = compute_map_moments(
            map_values, grid.positions_xyz_m, background_percentile
        )
        response_map = compute_point_spread_function(
            np.array(
                [
                    moments.centroid_xy_m[0],
                    moments.centroid_xy_m[1],
                    configuration.geometry.source_height_m,
                ]
            ),
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
        response_covariance_m2 = compute_map_moments(
            response_map, grid.positions_xyz_m, background_percentile
        ).covariance_xy_m2
        extent_response_covariance_m2 = compute_map_moments(
            response_map, grid.positions_xyz_m, extent_background_percentile
        ).covariance_xy_m2
        extent_response_semi_axis_m = float(
            compute_point_spread_semi_axes_m(extent_response_covariance_m2)[0]
        )
        truth = extract_fire_front_truth(
            fire_state.current_time_s,
            np.asarray(sources.source_positions_xy_m, dtype=np.float64),
            ignition_position_xy_m,
            cell_area_m2,
            characterization.extent.shape_factor,
        )
        extent = estimate_fire_extent(
            fire_state.current_time_s,
            map_values,
            grid.positions_xyz_m,
            extent_background_percentile,
            extent_response_covariance_m2,
            characterization.extent.shape_factor,
        )
        skewness_bearing = estimate_bearing_from_skewness(
            map_values,
            grid.positions_xyz_m,
            background_percentile,
            characterization.bearing.direction_count,
        )
        principal_axis_bearing: BearingEstimate | None
        try:
            principal_axis_bearing = estimate_bearing_from_principal_axis(
                map_values,
                grid.positions_xyz_m,
                ignition_position_xy_m,
                background_percentile,
                response_covariance_m2,
            )
        except ValueError:
            principal_axis_bearing = None

        frames.append(
            CharacterizationFrame(
                time_s=fire_state.current_time_s,
                map_values=map_values,
                truth=truth,
                response_covariance_m2=extent_response_covariance_m2,
                extent=extent,
                skewness_bearing=skewness_bearing,
                principal_axis_bearing=principal_axis_bearing,
                centroid_offset_bearing=None,
                map_centroid_xy_m=moments.centroid_xy_m,
                response_map=response_map,
            )
        )
        response_semi_axis_m = extent_response_semi_axis_m
        active_count = int(truth.active_cell_positions_xy_m.shape[0])
        print(
            f"  t = {fire_state.current_time_s:5.1f} s  active {active_count:4d}  "
            f"true a {truth.principal_semi_axes_m[0]:5.2f} m  "
            f"response a {response_semi_axis_m:5.2f} m  "
            f"est a {extent.semi_axis_major_m:5.2f} m "
            f"{'resolved' if extent.is_resolved else 'UNRESOLVED'}  "
            f"skew bearing {np.rad2deg(skewness_bearing.bearing_rad):7.1f} deg  "
            f"true {np.rad2deg(truth.bearing_rad):7.1f} deg"
        )
    return (
        frames,
        receiver_positions_xyz_m,
        grid.positions_xyz_m,
        ignition_position_xy_m,
    )


def attach_centroid_offset_bearings(
    frames: list[CharacterizationFrame],
    candidate_positions_xyz_m: Float64Array,
    estimated_ignition_xy_m: Float64Array,
    background_percentile: float,
    minimum_offset_m: float,
) -> list[CharacterizationFrame]:
    """Fills in each frame's bearing from ignition to its own centroid.

    A second pass rather than part of the first, because the ignition point is
    itself estimated from the earliest frame and is not known while that frame
    is being built. Nothing here reads the true ignition position.

    Args:
        frames: Every window's result, in time order.
        candidate_positions_xyz_m: Cell centres shared by all frames.
        estimated_ignition_xy_m: Ignition point from the first frame.
        background_percentile: Percentile of a frame treated as background.
        minimum_offset_m: Below this the centroid has not left the origin.

    Returns:
        The frames with their offset bearings attached.
    """
    attached: list[CharacterizationFrame] = []
    for frame in frames:
        bearing: BearingEstimate | None
        try:
            bearing = estimate_bearing_from_centroid_offset(
                frame.map_values,
                candidate_positions_xyz_m,
                estimated_ignition_xy_m,
                background_percentile,
                minimum_offset_m,
            )
        except ValueError:
            bearing = None
        attached.append(replace(frame, centroid_offset_bearing=bearing))
    return attached


def build_sequence(
    frames: list[CharacterizationFrame],
    candidate_positions_xyz_m: Float64Array,
    grid_spacing_m: float,
    domain_extent_x_m: float,
    domain_extent_y_m: float,
) -> SteeredResponsePowerSequence:
    """Collects the per-window maps into the sequence the drift estimator reads.

    Args:
        frames: Every window's result.
        candidate_positions_xyz_m: Cell centres shared by all frames.
        grid_spacing_m: Imaging cell side length, in metres.
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.

    Returns:
        The sequence.
    """
    return SteeredResponsePowerSequence(
        maps=np.stack([frame.map_values for frame in frames]),
        times_s=np.array([frame.time_s for frame in frames], dtype=np.float64),
        candidate_positions_xyz_m=candidate_positions_xyz_m,
        grid_shape=compute_imaging_grid_shape(
            domain_extent_x_m, domain_extent_y_m, grid_spacing_m
        ),
        grid_spacing_m=grid_spacing_m,
    )


def build_bearing_report(
    frames: list[CharacterizationFrame],
    drift: BearingEstimate | None,
    fused: BearingEstimate,
    true_bearing_rad: float,
) -> dict[str, Any]:
    """Scores every bearing method against the direction the fire took.

    Args:
        frames: Every window's result.
        drift: The centroid-drift estimate, or None when it refused.
        fused: The fused estimate.
        true_bearing_rad: The direction the fire actually travelled.

    Returns:
        A JSON-serialisable mapping.
    """
    settled = [frame for frame in frames if frame.time_s >= SETTLED_TIME_S]
    skewness_rad = np.array([frame.skewness_bearing.bearing_rad for frame in settled])
    truth_rad = np.array([frame.truth.bearing_rad for frame in settled])
    principal_frames = [
        frame for frame in settled if frame.principal_axis_bearing is not None
    ]
    principal_rad = np.array(
        [
            frame.principal_axis_bearing.bearing_rad
            for frame in principal_frames
            if frame.principal_axis_bearing is not None
        ]
    )
    principal_truth_rad = np.array(
        [frame.truth.bearing_rad for frame in principal_frames]
    )
    offset_frames = [
        frame for frame in settled if frame.centroid_offset_bearing is not None
    ]
    offset_rad = np.array(
        [
            frame.centroid_offset_bearing.bearing_rad
            for frame in offset_frames
            if frame.centroid_offset_bearing is not None
        ]
    )
    offset_truth_rad = np.array([frame.truth.bearing_rad for frame in offset_frames])
    return {
        "settled_after_s": SETTLED_TIME_S,
        "true_bearing_rad": true_bearing_rad,
        "true_bearing_deg": float(np.rad2deg(true_bearing_rad)),
        "centroid_drift": None
        if drift is None
        else {
            "bearing_deg": float(np.rad2deg(drift.bearing_rad)),
            "circular_standard_deviation_deg": float(
                np.rad2deg(drift.circular_standard_deviation_rad)
            ),
            "error_deg": float(
                np.rad2deg(
                    compute_circular_error_rad(drift.bearing_rad, true_bearing_rad)
                )
            ),
            "frames_used": drift.frames_used,
        },
        "skewness_per_frame": summarize_bearing_errors(
            compute_bearing_error_series(skewness_rad, truth_rad)
        ),
        "centroid_offset_per_frame": summarize_bearing_errors(
            compute_bearing_error_series(offset_rad, offset_truth_rad)
        )
        if offset_rad.size
        else None,
        "principal_axis_per_frame": summarize_bearing_errors(
            compute_bearing_error_series(principal_rad, principal_truth_rad)
        )
        if principal_rad.size
        else None,
        "fused": {
            "bearing_deg": float(np.rad2deg(fused.bearing_rad)),
            "circular_standard_deviation_deg": float(
                np.rad2deg(fused.circular_standard_deviation_rad)
            ),
            "resultant_length": fused.resultant_length,
            "error_deg": float(
                np.rad2deg(
                    compute_circular_error_rad(fused.bearing_rad, true_bearing_rad)
                )
            ),
        },
        "drift_and_skewness_agreement_fraction": float(
            compute_agreement_fraction(
                np.full(skewness_rad.size, drift.bearing_rad),
                skewness_rad,
                AGREEMENT_TOLERANCE_DEG,
            )
        )
        if drift is not None and skewness_rad.size
        else None,
    }


def build_spread_report(
    frames: list[CharacterizationFrame], rate: RateOfSpreadEstimate | None
) -> dict[str, Any]:
    """Scores the extent series and the rate of spread against truth.

    Args:
        frames: Every window's result.
        rate: What the rate stage returned, or None when it refused.

    Returns:
        A JSON-serialisable mapping.
    """
    times_s = np.array([frame.time_s for frame in frames])
    true_head_m = np.array([frame.truth.head_distance_m for frame in frames])
    true_back_m = np.array([frame.truth.back_distance_m for frame in frames])
    resolved = [frame for frame in frames if frame.extent.is_resolved]
    extent_errors = (
        compute_extent_error_series(
            np.array([frame.extent.semi_axis_major_m for frame in resolved]),
            np.array([frame.truth.principal_semi_axes_m[0] for frame in resolved]),
        )
        if resolved
        else np.empty(0)
    )
    honest_flags = [
        is_flag_honest(
            frame.extent.is_resolved,
            float(frame.truth.principal_semi_axes_m[0]),
            float(compute_point_spread_semi_axes_m(frame.response_covariance_m2)[0]),
        )
        for frame in frames
    ]
    true_head_rate = compute_true_rate_of_spread_m_per_s(times_s, true_head_m)
    true_back_rate = compute_true_rate_of_spread_m_per_s(times_s, true_back_m)
    return {
        "resolved_frame_count": len(resolved),
        "frame_count": len(frames),
        "every_unresolved_flag_is_honest": bool(all(honest_flags)),
        "median_extent_relative_error": float(np.nanmedian(extent_errors))
        if extent_errors.size
        else None,
        "maximum_extent_relative_error": float(np.nanmax(extent_errors))
        if extent_errors.size
        else None,
        "head": compute_rate_of_spread_error(
            rate.head_rate_of_spread_m_per_s, true_head_rate
        )
        if rate is not None
        else {"true_m_per_s": true_head_rate},
        "back": compute_rate_of_spread_error(
            rate.back_rate_of_spread_m_per_s, true_back_rate
        )
        if rate is not None
        else {"true_m_per_s": true_back_rate},
        "estimate": rate.to_dict() if rate is not None else None,
    }


def main() -> None:
    """Runs both tiers, prints what they found and writes the report."""
    arguments = parse_arguments()
    forward_config = load_forward_simulation_configuration(arguments.configs)
    configuration = load_simulation_configuration(arguments.configs)
    characterization = load_fire_characterization_configuration(
        arguments.configs / FIRE_CHARACTERIZATION_FILENAME
    )
    background_percentile = (
        arguments.background_percentile
        if arguments.background_percentile is not None
        else characterization.imaging_map.background_percentile
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
        f"{forward_config.experiment.name}: bearing and rate of spread from "
        f"{configuration.geometry.receiver_layout.count} receivers, "
        f"{observation_count} windows, imaging "
        f"{characterization.imaging_map.pairwise_combinator}/"
        f"{characterization.imaging_map.pooling} at beta="
        f"{characterization.imaging_map.phase_transform_exponent}, "
        f"percentiles {background_percentile:.0f} for centroids, "
        f"{characterization.imaging_map.extent_background_percentile:.0f} for extent"
    )
    frames, receivers_xyz_m, candidates_xyz_m, ignition_xy_m = build_frames(
        forward_config,
        configuration,
        characterization,
        background_percentile,
        characterization.imaging_map.extent_background_percentile,
        observation_count,
    )
    domain = configuration.geometry.domain
    sequence = build_sequence(
        frames,
        candidates_xyz_m,
        characterization.imaging_map.grid_spacing_m,
        domain.size_x_m,
        domain.size_y_m,
    )

    estimated_ignition_xy_m = estimate_ignition_position_xy_m(
        sequence, background_percentile, None
    )
    frames = attach_centroid_offset_bearings(
        frames,
        candidates_xyz_m,
        estimated_ignition_xy_m,
        background_percentile,
        characterization.bearing.minimum_centroid_displacement_m,
    )
    print(
        f"  ignition estimated at "
        f"({estimated_ignition_xy_m[0]:.2f}, {estimated_ignition_xy_m[1]:.2f}) m, "
        f"truth ({ignition_xy_m[0]:.2f}, {ignition_xy_m[1]:.2f}) m, "
        f"error {float(np.linalg.norm(estimated_ignition_xy_m - ignition_xy_m)):.2f} m"
    )

    drift: BearingEstimate | None
    try:
        drift = estimate_bearing_from_centroid_drift(
            sequence,
            background_percentile,
            None,
            characterization.bearing.minimum_centroid_displacement_m,
            characterization.bearing.bootstrap_resample_count,
            configuration.forward_model.random_seed,
        )
    except ValueError as refusal:
        print(f"  centroid drift refused: {refusal}")
        drift = None

    last = frames[-1]
    candidates = [last.skewness_bearing]
    if drift is not None:
        candidates.append(drift)
    if last.principal_axis_bearing is not None:
        candidates.append(last.principal_axis_bearing)
    if last.centroid_offset_bearing is not None:
        candidates.append(last.centroid_offset_bearing)
    fused = fuse_bearing_estimates(candidates, characterization.bearing.fusion_weights)

    rate: RateOfSpreadEstimate | None
    try:
        rate = estimate_rate_of_spread(
            [frame.extent for frame in frames],
            fused.bearing_rad,
            characterization.extent.minimum_resolved_frames,
            characterization.bearing.bootstrap_resample_count,
            configuration.forward_model.random_seed,
        )
    except ValueError as refusal:
        print(f"  rate of spread refused: {refusal}")
        rate = None

    true_bearing_rad = float(frames[-1].truth.bearing_rad)
    bearing_report = build_bearing_report(frames, drift, fused, true_bearing_rad)
    spread_report = build_spread_report(frames, rate)

    print("\nBEARING")
    print(f"  true                {np.rad2deg(true_bearing_rad):8.2f} deg")
    if drift is not None:
        print(
            f"  centroid drift      {np.rad2deg(drift.bearing_rad):8.2f} deg   "
            f"error {bearing_report['centroid_drift']['error_deg']:+7.2f} deg"
        )
    offset = bearing_report["centroid_offset_per_frame"]
    if offset is not None:
        print(
            f"  centroid offset     median error "
            f"{offset['median_absolute_error_deg']:6.2f} deg, "
            f"90th {offset['ninetieth_percentile_absolute_error_deg']:6.2f} deg, "
            f"within 10 deg {offset['fraction_within_10_deg']:.0%}"
        )
    skew = bearing_report["skewness_per_frame"]
    median_deg = skew["median_absolute_error_deg"]
    ninetieth_deg = skew["ninetieth_percentile_absolute_error_deg"]
    if median_deg is None:
        print(
            f"  skewness per frame  no frame settled past {SETTLED_TIME_S:.0f} s, "
            "so there is nothing to summarise"
        )
    else:
        print(
            f"  skewness per frame  median error {median_deg:6.2f} deg, "
            f"90th {ninetieth_deg:6.2f} deg, "
            f"within 10 deg {skew['fraction_within_10_deg']:.0%}"
        )
    print(
        f"  fused               {np.rad2deg(fused.bearing_rad):8.2f} deg   "
        f"error {bearing_report['fused']['error_deg']:+7.2f} deg"
    )

    print("\nRATE OF SPREAD")
    if rate is None:
        print("  not estimated")
    else:
        head = spread_report["head"]
        back = spread_report["back"]
        print(
            f"  head   {head['estimated_m_per_s']:.4f} m/s  true "
            f"{head['true_m_per_s']:.4f}  error {head['relative_error']:.1%}"
        )
        print(
            f"  back   {back['estimated_m_per_s']:.4f} m/s  true "
            f"{back['true_m_per_s']:.4f}"
        )
        print(
            f"  closure residual {rate.closure_residual_fraction:.1%} of head, "
            f"resolved frames {rate.frames_used}/{len(frames)}"
        )

    figure_directory = FIGURE_ROOT / arguments.configs.name
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure = plot_fire_characterization(
        np.array([frame.time_s for frame in frames]),
        np.array([np.rad2deg(frame.skewness_bearing.bearing_rad) for frame in frames]),
        np.array([np.rad2deg(frame.truth.bearing_rad) for frame in frames]),
        float(np.rad2deg(fused.bearing_rad)),
        np.array([frame.extent.semi_axis_major_m for frame in frames]),
        np.array([frame.truth.principal_semi_axes_m[0] for frame in frames]),
        np.array([frame.extent.is_resolved for frame in frames]),
        np.array(
            [
                compute_point_spread_semi_axes_m(frame.response_covariance_m2)[0]
                for frame in frames
            ]
        ),
    )
    figure_path = figure_directory / "tier12_fire_characterization.svg"
    figure.savefig(figure_path, bbox_inches="tight")

    metrics_path = METRICS_ROOT / "tier12_fire_characterization.json"
    write_metrics_document(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration_directory": str(arguments.configs),
            "receiver_count": int(receivers_xyz_m.shape[0]),
            "background_percentile": background_percentile,
            "extent_background_percentile": (
                characterization.imaging_map.extent_background_percentile
            ),
            "imaging_map": {
                "phase_transform_exponent": (
                    characterization.imaging_map.phase_transform_exponent
                ),
                "pairwise_combinator": (
                    characterization.imaging_map.pairwise_combinator
                ),
                "pooling": characterization.imaging_map.pooling,
                "grid_spacing_m": characterization.imaging_map.grid_spacing_m,
            },
            "ignition_position_xy_m": ignition_xy_m.tolist(),
            "bearing": bearing_report,
            "rate_of_spread": spread_report,
            "frames": [
                {
                    "time_s": frame.time_s,
                    "active_cell_count": int(
                        frame.truth.active_cell_positions_xy_m.shape[0]
                    ),
                    "true_semi_axis_major_m": float(
                        frame.truth.principal_semi_axes_m[0]
                    ),
                    "response_semi_axis_major_m": float(
                        compute_point_spread_semi_axes_m(frame.response_covariance_m2)[
                            0
                        ]
                    ),
                    "estimated_semi_axis_major_m": frame.extent.semi_axis_major_m,
                    "is_resolved": frame.extent.is_resolved,
                    "true_bearing_deg": float(np.rad2deg(frame.truth.bearing_rad)),
                    "skewness_bearing_deg": float(
                        np.rad2deg(frame.skewness_bearing.bearing_rad)
                    ),
                }
                for frame in frames
            ],
        },
        metrics_path,
    )
    print(f"\nfigure  {figure_path}")
    print(f"metrics {metrics_path}")


if __name__ == "__main__":
    main()
