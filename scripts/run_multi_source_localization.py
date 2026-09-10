"""Locates several concurrent acoustic sources from one receiver array.

Usage:
    uv run python scripts/run_multi_source_localization.py
    uv run python scripts/run_multi_source_localization.py --configs configs/m1

Zero domain logic lives here. The script gives each configured source its own
recording excerpt, renders the mixture through the forward channel to the
configured receiver layout, hands the receiver waveforms to the estimator, and
scores what comes back against the truth it never showed the estimator.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from src.audio.excerpt_coherence import (
    compute_maximum_off_diagonal_coherence,
    compute_pairwise_excerpt_coherence_matrix,
)
from src.audio.octave_band_filter import (
    build_fractional_octave_centre_frequencies_hz,
    compute_fractional_octave_band_edges_hz,
)
from src.audio.source_excerpt_selector import (
    SourceExcerpt,
    list_pool_recordings,
    select_source_excerpts,
)
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    SimulationConfiguration,
    load_experiment_configuration,
    load_simulation_configuration,
)
from src.config.source_scene_configuration import (
    collect_source_amplitude_scales,
    stack_source_positions_xy_m,
)
from src.spark.acoustic.free_field_propagation import (
    render_multi_source_receiver_signals,
)
from src.spark.acoustic.receiver_noise import add_white_noise_to_receiver_signals
from src.spark.acoustic.receiver_placement import place_receivers_from_layout
from src.spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from src.spark.inverse.joint_position_refinement import (
    LevelRefinedPosition,
    refine_position_with_band_levels,
)
from src.spark.inverse.multiple_source_estimator import (
    MultiSourceLocalization,
    localize_multiple_sources,
)
from src.spark.inverse.windowed_position_clustering import (
    WindowedClustering,
    estimate_positions_by_windowed_clustering,
)
from src.utils.array_types import Float64Array
from src.utils.io.metrics_writer import write_metrics_document
from src.utils.metrics.localization_metrics import (
    build_localization_metric_record,
    match_estimated_to_true_sources,
)
from src.utils.visualization.steered_response_power_plotter import (
    plot_steered_response_power_map,
    plot_windowed_position_clusters,
)

FIGURE_ROOT: Path = Path("results/figures")
FIGURE_FORMAT: str = "svg"


@dataclass(frozen=True)
class MultiSourceRun:
    """Everything one run produced, before it is scored or printed.

    Attributes:
        excerpts: The excerpt each source was given.
        receiver_positions_xyz_m: The resolved receiver layout.
        receiver_signals: The rendered mixture, one channel per receiver.
        sample_rate_hz: Sample rate in hertz.
        localization: What the sequential estimator found.
        clustering: What the independent windowed cross-check found, or None.
        refined_positions: Level-fused positions, one per located source.
        achieved_excerpt_coherence: Worst mutual coherence between the excerpts
            the sources were actually given.
    """

    excerpts: list[SourceExcerpt]
    receiver_positions_xyz_m: Float64Array
    receiver_signals: Float64Array
    sample_rate_hz: int
    localization: MultiSourceLocalization
    clustering: WindowedClustering | None
    refined_positions: list[LevelRefinedPosition]
    achieved_excerpt_coherence: float


def parse_arguments() -> argparse.Namespace:
    """Parses the only argument this script takes.

    Returns:
        The parsed arguments, carrying the configuration directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    return parser.parse_args()


def build_band_arrays(
    configuration: SimulationConfiguration, sample_rate_hz: int
) -> tuple[Float64Array, Float64Array]:
    """Resolves the level-stage band centres and their filter edges.

    Args:
        configuration: The whole run configuration.
        sample_rate_hz: Sample rate in hertz.

    Returns:
        `(centre_frequencies_hz, band_edges_hz)`, shapes `(n_bands,)` and
        `(n_bands, 2)`.
    """
    bands = configuration.localization.multi_source.bands
    centres_hz = build_fractional_octave_centre_frequencies_hz(
        bands.lowest_centre_hz, bands.highest_centre_hz, bands.fraction_denominator
    )
    edges_hz = np.array(
        [
            compute_fractional_octave_band_edges_hz(
                centre_hz,
                bands.fraction_denominator,
                sample_rate_hz,
                bands.maximum_edge_fraction_of_nyquist,
            )
            for centre_hz in centres_hz
        ],
        dtype=np.float64,
    )
    return np.asarray(centres_hz, dtype=np.float64), edges_hz


def render_scene(
    configuration: SimulationConfiguration,
    excerpts: list[SourceExcerpt],
    receiver_positions_xyz_m: Float64Array,
    sample_rate_hz: int,
) -> Float64Array:
    """Renders every configured source to every receiver and adds sensor noise.

    Args:
        configuration: The whole run configuration.
        excerpts: One excerpt per source.
        receiver_positions_xyz_m: The resolved layout, shape `(n_receivers, 3)`.
        sample_rate_hz: Sample rate in hertz.

    Returns:
        Receiver channels of shape `(n_receivers, n_samples)`.
    """
    forward_model = configuration.forward_model
    receiver_signals = render_multi_source_receiver_signals(
        [excerpt.samples for excerpt in excerpts],
        configuration.geometry.concurrent_source_positions_xyz_m,
        collect_source_amplitude_scales(configuration.geometry.concurrent_sources),
        receiver_positions_xyz_m,
        sample_rate_hz,
        configuration.atmosphere,
        forward_model.propagation.reference_distance_m,
    )
    signal_to_noise_ratio_db = (
        forward_model.receiver_noise.requested_signal_to_noise_ratio_db
    )
    if signal_to_noise_ratio_db is None:
        return receiver_signals
    return add_white_noise_to_receiver_signals(
        receiver_signals,
        signal_to_noise_ratio_db,
        np.random.default_rng(forward_model.random_seed),
    )


def refine_located_sources(
    configuration: SimulationConfiguration,
    localization: MultiSourceLocalization,
    receiver_signals: Float64Array,
    sample_rate_hz: int,
) -> list[LevelRefinedPosition]:
    """Fuses band levels into each delay-only position.

    Args:
        configuration: The whole run configuration.
        localization: What the sequential estimator found.
        receiver_signals: The rendered mixture.
        sample_rate_hz: Sample rate in hertz.

    Returns:
        One refined position per located source, empty when the level terms are
        switched off.
    """
    multi_source = configuration.localization.multi_source
    if not multi_source.refinement.use_band_level_terms:
        return []
    centres_hz, edges_hz = build_band_arrays(configuration, sample_rate_hz)
    absorption_db_per_m = compute_absorption_coefficients_db_per_m(
        centres_hz, configuration.atmosphere
    )
    window_sample_count = round(
        multi_source.correlation.window_duration_s * sample_rate_hz
    )
    return [
        refine_position_with_band_levels(
            source.map_position_xyz_m,
            source.pair_delays_s,
            source.pair_delay_variances_s2,
            receiver_signals,
            localization.receiver_positions_xyz_m,
            localization.receiver_pairs,
            sample_rate_hz,
            centres_hz,
            edges_hz,
            absorption_db_per_m,
            localization.speed_of_sound_m_per_s,
            window_sample_count,
            multi_source.correlation.window_overlap,
            multi_source.refinement,
        )
        for source in localization.sources
    ]


def execute_run(configuration: SimulationConfiguration) -> MultiSourceRun:
    """Selects excerpts, renders the scene and runs every estimator on it.

    Args:
        configuration: The whole run configuration.

    Returns:
        Everything the run produced.
    """
    excerpts_configuration = configuration.data.excerpts
    excerpts = select_source_excerpts(
        list_pool_recordings(
            excerpts_configuration.recording_directory,
            excerpts_configuration.recording_filenames,
        ),
        len(configuration.geometry.concurrent_sources),
        configuration.data.segmentation.clip_duration_s,
        excerpts_configuration.assignment_policy,
        excerpts_configuration.maximum_permitted_excerpt_coherence,
        excerpts_configuration.selection_seed,
        configuration.data.recording.as_mono,
    )
    sample_rate_hz = excerpts[0].sample_rate_hz
    receiver_positions_xyz_m = place_receivers_from_layout(
        configuration.geometry.receiver_layout,
        configuration.geometry.domain.size_x_m,
        configuration.geometry.domain.size_y_m,
    )
    receiver_signals = render_scene(
        configuration, excerpts, receiver_positions_xyz_m, sample_rate_hz
    )

    multi_source = configuration.localization.multi_source
    localization = localize_multiple_sources(
        receiver_signals,
        receiver_positions_xyz_m,
        sample_rate_hz,
        configuration.geometry.domain.size_x_m,
        configuration.geometry.domain.size_y_m,
        configuration.atmosphere,
        multi_source,
    )
    clustering = (
        estimate_positions_by_windowed_clustering(
            receiver_signals,
            receiver_positions_xyz_m,
            sample_rate_hz,
            configuration.geometry.domain.size_x_m,
            configuration.geometry.domain.size_y_m,
            configuration.atmosphere,
            multi_source,
        )
        if multi_source.clustering.enabled
        else None
    )
    return MultiSourceRun(
        excerpts=excerpts,
        receiver_positions_xyz_m=receiver_positions_xyz_m,
        receiver_signals=receiver_signals,
        sample_rate_hz=sample_rate_hz,
        localization=localization,
        clustering=clustering,
        refined_positions=refine_located_sources(
            configuration, localization, receiver_signals, sample_rate_hz
        ),
        achieved_excerpt_coherence=compute_maximum_off_diagonal_coherence(
            compute_pairwise_excerpt_coherence_matrix(
                [excerpt.samples for excerpt in excerpts]
            )
        ),
    )


def stack_estimated_positions_xy_m(run: MultiSourceRun) -> Float64Array:
    """Collects the delay-only estimates into one array.

    Args:
        run: The completed run.

    Returns:
        Positions of shape `(n_estimated, 2)`, empty when nothing was found.
    """
    if not run.localization.sources:
        return np.empty((0, 2), dtype=np.float64)
    return np.stack(
        [source.position.position_xy_m for source in run.localization.sources]
    )


def stack_refined_positions_xy_m(run: MultiSourceRun) -> Float64Array:
    """Collects the level-fused estimates into one array.

    Args:
        run: The completed run.

    Returns:
        Positions of shape `(n_refined, 2)`, empty when refinement was skipped.
    """
    if not run.refined_positions:
        return np.empty((0, 2), dtype=np.float64)
    return np.stack([refined.position_xy_m for refined in run.refined_positions])


def compute_radial_semi_axis_m(position_covariance_m2: Float64Array) -> float:
    """Reports the larger semi-axis of a one-sigma error ellipse.

    Args:
        position_covariance_m2: Two-by-two covariance in metres squared.

    Returns:
        The major semi-axis, in metres.
    """
    eigenvalues = np.linalg.eigvalsh(
        np.asarray(position_covariance_m2, dtype=np.float64)
    )
    return float(np.sqrt(max(float(np.max(eigenvalues)), 0.0)))


def print_run_report(
    configuration: SimulationConfiguration, run: MultiSourceRun
) -> None:
    """Prints the scene, what was found and how well it matched the truth.

    Args:
        configuration: The whole run configuration.
        run: The completed run.
    """
    geometry = configuration.geometry
    true_positions_xy_m = stack_source_positions_xy_m(geometry.concurrent_sources)
    noise = configuration.forward_model.receiver_noise
    multi_source = configuration.localization.multi_source

    print(f"configs      : {configuration.configuration_directory}")
    print(
        f"receivers    : {geometry.receiver_layout.layout}, "
        f"{run.receiver_positions_xyz_m.shape[0]} at "
        f"{geometry.receiver_layout.height_m} m"
    )
    for position_xyz_m in run.receiver_positions_xyz_m:
        print(
            f"               ({position_xyz_m[0]:>7.2f}, {position_xyz_m[1]:>7.2f}) m"
        )
    print(f"sources      : {len(geometry.concurrent_sources)}")
    for source_index, source in enumerate(geometry.concurrent_sources):
        print(
            f"               {source_index}: "
            f"({source.position_xy_m[0]:.1f}, {source.position_xy_m[1]:.1f}) m, "
            f"amplitude {source.amplitude_scale:.2f}, "
            f"{run.excerpts[source_index].recording_path.name}"
        )
    print(
        f"clip         : {configuration.data.segmentation.clip_duration_s:.0f} s at "
        f"{run.sample_rate_hz} Hz"
    )
    print(
        f"excerpts     : mutual coherence {run.achieved_excerpt_coherence:.3f} "
        f"| permitted "
        f"{configuration.data.excerpts.maximum_permitted_excerpt_coherence:.3f}"
    )
    print(
        "noise        : "
        + (
            f"white, {noise.signal_to_noise_ratio_db} dB SNR"
            if noise.enabled
            else "none"
        )
    )
    print(
        f"search       : {multi_source.steered_response_power.pooling} pooling, "
        f"{multi_source.steered_response_power.pairwise_combinator} combinator, "
        f"beta {multi_source.correlation.phase_transform_exponent}"
    )
    print(f"deflation    : {multi_source.deflation.method}")
    print()

    print(
        f"K_hat = {run.localization.estimated_source_count} "
        f"(K = {len(geometry.concurrent_sources)}), "
        f"stopped on {run.localization.stop_reason}"
    )
    print(
        "  src     x_hat     y_hat  prominence  residual  chi2_nu   ellipse"
        "     x_lvl     y_lvl  ellipse_lvl"
    )
    refined_positions_xy_m = stack_refined_positions_xy_m(run)
    for source_index, located in enumerate(run.localization.sources):
        level_columns = "         -         -            -"
        if source_index < refined_positions_xy_m.shape[0]:
            refined = run.refined_positions[source_index]
            level_columns = (
                f"{refined.position_xy_m[0]:>10.2f}"
                f"{refined.position_xy_m[1]:>10.2f}"
                f"{compute_radial_semi_axis_m(refined.position_covariance_m2):>13.2f}"
            )
        delay_only_semi_axis_m = compute_radial_semi_axis_m(
            located.position.position_covariance_m2
        )
        print(
            f"  {source_index:>3}"
            f"{located.position.position_xy_m[0]:>10.2f}"
            f"{located.position.position_xy_m[1]:>10.2f}"
            f"{located.map_peak_to_median_ratio:>12.2f}"
            f"{located.residual_energy_ratio:>10.3f}"
            f"{located.position.reduced_chi_square:>9.2f}"
            f"{delay_only_semi_axis_m:>10.2f}"
            f"{level_columns}"
        )

    assignment = match_estimated_to_true_sources(
        true_positions_xy_m, stack_estimated_positions_xy_m(run)
    )
    print()
    print("matched against truth")
    for true_index, estimated_index, error_m in zip(
        assignment.matched_true_indices,
        assignment.matched_estimated_indices,
        assignment.matched_errors_m,
        strict=True,
    ):
        print(
            f"  true {int(true_index)} -> estimate {int(estimated_index)}: "
            f"{float(error_m):.2f} m"
        )
    print(
        f"  detections {assignment.detection_count} | "
        f"missed {assignment.missed_count} | "
        f"false alarms {assignment.false_alarm_count}"
    )
    if run.clustering is not None:
        print()
        print(
            f"windowed clustering cross-check: K_hat = "
            f"{run.clustering.estimated_source_count} from "
            f"{run.clustering.window_maxima_xy_m.shape[0]} windows"
        )
        for cluster_index, cluster in enumerate(run.clustering.clusters):
            print(
                f"  cluster {cluster_index}: "
                f"({cluster.centroid_xy_m[0]:.2f}, {cluster.centroid_xy_m[1]:.2f}) m, "
                f"spread {cluster.spread_m:.2f} m, {cluster.window_count} windows"
            )
    print()


def save_figure(figure: Figure, figure_directory: Path, name: str) -> Path:
    """Writes one figure under the run's figure directory.

    Args:
        figure: The figure to write.
        figure_directory: Directory to write into, created if absent.
        name: File stem.

    Returns:
        The path written to.
    """
    figure_directory.mkdir(parents=True, exist_ok=True)
    path = figure_directory / f"{name}.{FIGURE_FORMAT}"
    figure.savefig(path, format=FIGURE_FORMAT, bbox_inches="tight")
    return path


def export_figures(
    configuration: SimulationConfiguration, run: MultiSourceRun, figure_directory: Path
) -> list[Path]:
    """Draws and saves the search map and the clustering cross-check.

    Args:
        configuration: The whole run configuration.
        run: The completed run.
        figure_directory: Directory to write into.

    Returns:
        The paths written.
    """
    true_positions_xy_m = stack_source_positions_xy_m(
        configuration.geometry.concurrent_sources
    )
    paths = [
        save_figure(
            plot_steered_response_power_map(
                run.localization.first_round_map,
                run.localization.first_round_positions_xyz_m,
                run.receiver_positions_xyz_m,
                true_positions_xy_m,
                stack_estimated_positions_xy_m(run),
                [
                    source.position.position_covariance_m2
                    for source in run.localization.sources
                ],
                "steered response power, first search round",
            ),
            figure_directory,
            "steered_response_power_map",
        )
    ]
    if run.clustering is not None:
        paths.append(
            save_figure(
                plot_windowed_position_clusters(
                    run.clustering.window_maxima_xy_m,
                    np.stack(
                        [cluster.centroid_xy_m for cluster in run.clustering.clusters]
                    )
                    if run.clustering.clusters
                    else np.empty((0, 2), dtype=np.float64),
                    true_positions_xy_m,
                    run.receiver_positions_xyz_m,
                ),
                figure_directory,
                "windowed_position_clusters",
            )
        )
    return paths


def build_metrics_document(
    configuration: SimulationConfiguration, run: MultiSourceRun
) -> dict[str, object]:
    """Builds the whole metrics document, settings included.

    Args:
        configuration: The whole run configuration.
        run: The completed run.

    Returns:
        A JSON-serializable document describing the run and its results.
    """
    multi_source = configuration.localization.multi_source
    true_positions_xy_m = stack_source_positions_xy_m(
        configuration.geometry.concurrent_sources
    )
    estimated_positions_xy_m = stack_estimated_positions_xy_m(run)
    refined_positions_xy_m = stack_refined_positions_xy_m(run)
    return {
        "configuration_directory": str(configuration.configuration_directory),
        "recording_directory": str(configuration.data.excerpts.recording_directory),
        "source_recordings": [excerpt.recording_path.name for excerpt in run.excerpts],
        "assignment_policy": configuration.data.excerpts.assignment_policy,
        "maximum_permitted_excerpt_coherence": (
            configuration.data.excerpts.maximum_permitted_excerpt_coherence
        ),
        "achieved_excerpt_coherence": run.achieved_excerpt_coherence,
        "sample_rate_hz": run.sample_rate_hz,
        "clip_duration_s": configuration.data.segmentation.clip_duration_s,
        "receiver_layout": configuration.geometry.receiver_layout.to_dict(),
        "receiver_positions_xyz_m": run.receiver_positions_xyz_m,
        "true_source_positions_xy_m": true_positions_xy_m,
        "true_source_amplitude_scales": collect_source_amplitude_scales(
            configuration.geometry.concurrent_sources
        ),
        "air_temperature_celsius": configuration.atmosphere.air_temperature_celsius,
        "relative_humidity_percent": configuration.atmosphere.relative_humidity_percent,
        "pressure_kpa": configuration.atmosphere.pressure_kpa,
        "signal_to_noise_ratio_db": (
            configuration.forward_model.receiver_noise.requested_signal_to_noise_ratio_db
        ),
        "random_seed": configuration.forward_model.random_seed,
        "phase_transform_exponent": multi_source.correlation.phase_transform_exponent,
        "pooling": multi_source.steered_response_power.pooling,
        "pairwise_combinator": (
            multi_source.steered_response_power.pairwise_combinator
        ),
        "deflation_method": multi_source.deflation.method,
        "stop_reason": run.localization.stop_reason,
        "estimated_source_count": run.localization.estimated_source_count,
        "estimated_positions_xy_m": estimated_positions_xy_m,
        "refined_positions_xy_m": refined_positions_xy_m,
        "delay_only_metrics": build_localization_metric_record(
            true_positions_xy_m,
            estimated_positions_xy_m,
            multi_source.metrics.optimal_subpattern_assignment_cutoff_m,
            multi_source.metrics.optimal_subpattern_assignment_order,
        ),
        "level_fused_metrics": build_localization_metric_record(
            true_positions_xy_m,
            refined_positions_xy_m,
            multi_source.metrics.optimal_subpattern_assignment_cutoff_m,
            multi_source.metrics.optimal_subpattern_assignment_order,
        ),
        "delay_only_ellipse_semi_major_m": [
            compute_radial_semi_axis_m(source.position.position_covariance_m2)
            for source in run.localization.sources
        ],
        "level_fused_ellipse_semi_major_m": [
            compute_radial_semi_axis_m(refined.position_covariance_m2)
            for refined in run.refined_positions
        ],
        "windowed_clustering_source_count": (
            run.clustering.estimated_source_count
            if run.clustering is not None
            else None
        ),
        "windowed_clustering_centroids_xy_m": (
            [cluster.centroid_xy_m for cluster in run.clustering.clusters]
            if run.clustering is not None
            else []
        ),
    }


def main() -> None:
    """Runs the configured multi-source scene end to end."""
    arguments = parse_arguments()
    configuration = load_simulation_configuration(arguments.configs)
    experiment = load_experiment_configuration(arguments.configs / "experiment.toml")

    run = execute_run(configuration)
    print_run_report(configuration, run)

    figure_paths = export_figures(
        configuration, run, FIGURE_ROOT / experiment.name / "multi_source"
    )
    metrics_path = write_metrics_document(
        build_metrics_document(configuration, run),
        configuration.data.output.metrics_path,
    )
    for path in figure_paths:
        print(f"figure written to {path}")
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
