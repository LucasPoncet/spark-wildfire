"""Locates a single acoustic source from two receivers, clip by clip.

Usage:
    uv run python scripts/run_single_source_localization.py
    uv run python scripts/run_single_source_localization.py --configs other_dir

Zero domain logic lives here. The script renders a recording through the forward
channel to the configured receivers, hands the two signals to the estimator, and
prints one table per source scenario. Every parameter comes from the configuration
directory, so a variant run is a copied directory, not an edited file.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio.recording_segmenter import segment_recording
from config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    SimulationConfiguration,
    load_simulation_configuration,
)
from spark.acoustic.free_field_propagation import render_receiver_signals
from spark.acoustic.receiver_noise import add_white_noise_to_receiver_signals
from spark.inverse.single_source_estimator import (
    SingleSourceEstimator,
    compute_error_ellipse_semi_axes_m,
)
from utils.array_types import Float64Array
from utils.io.audio_file_reader import read_audio_file
from utils.io.metrics_writer import write_metrics_document


@dataclass(frozen=True)
class ClipEstimate:
    """One clip's estimate against its known source position.

    Attributes:
        clip_index: Index of the clip within the recording.
        position_xy_m: Estimated position (x, y) in metres.
        error_m: Distance from the true position, in metres.
        path_difference_m: Range difference from the delay, in metres.
        geometric_level_difference_db: Fused level difference in decibels.
        reduced_chi_square: About 1 when the bands agree.
        ellipse_semi_major_m: One-sigma error ellipse major semi-axis.
        ellipse_semi_minor_m: One-sigma error ellipse minor semi-axis.
        is_near_singular: Whether the source sits near the bisector.
    """

    clip_index: int
    position_xy_m: Float64Array
    error_m: float
    path_difference_m: float
    geometric_level_difference_db: float
    reduced_chi_square: float
    ellipse_semi_major_m: float
    ellipse_semi_minor_m: float
    is_near_singular: bool


def parse_arguments() -> argparse.Namespace:
    """Parses the only argument this script takes.

    Returns:
        The parsed arguments, carrying the configuration directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGURATION_DIRECTORY)
    return parser.parse_args()


def estimate_over_clips(
    clips: Float64Array,
    sample_rate_hz: int,
    true_position_xy_m: Float64Array,
    estimator: SingleSourceEstimator,
    configuration: SimulationConfiguration,
) -> list[ClipEstimate]:
    """Estimates the source position once per clip.

    Args:
        clips: Source clips, shape (n_clips, n_samples).
        sample_rate_hz: Sample rate in hertz.
        true_position_xy_m: Ground-truth position, used only to report error.
        estimator: The configured two-receiver estimator.
        configuration: The whole run configuration.

    Returns:
        One estimate per clip.
    """
    forward_model = configuration.forward_model
    signal_to_noise_ratio_db = (
        forward_model.receiver_noise.requested_signal_to_noise_ratio_db
    )
    estimates: list[ClipEstimate] = []
    for clip_index, clip in enumerate(clips):
        receiver_signals = render_receiver_signals(
            clip,
            sample_rate_hz,
            true_position_xy_m,
            configuration.geometry.receiver_positions_xy_m,
            configuration.atmosphere,
            forward_model.propagation.reference_distance_m,
        )
        if signal_to_noise_ratio_db is not None:
            receiver_signals = add_white_noise_to_receiver_signals(
                receiver_signals,
                signal_to_noise_ratio_db,
                np.random.default_rng(forward_model.random_seed + clip_index),
            )
        localization = estimator.estimate(
            receiver_signals[0], receiver_signals[1], sample_rate_hz
        )
        semi_major_m, semi_minor_m, _ = compute_error_ellipse_semi_axes_m(
            localization.position_covariance_m2
        )
        estimates.append(
            ClipEstimate(
                clip_index=clip_index,
                position_xy_m=localization.position_xy_m,
                error_m=float(
                    np.linalg.norm(localization.position_xy_m - true_position_xy_m)
                ),
                path_difference_m=localization.path_difference_m,
                geometric_level_difference_db=localization.geometric_level_difference_db,
                reduced_chi_square=localization.reduced_chi_square,
                ellipse_semi_major_m=semi_major_m,
                ellipse_semi_minor_m=semi_minor_m,
                is_near_singular=localization.is_near_singular,
            )
        )
    return estimates


def format_scenario_table(
    true_position_xy_m: Float64Array, estimates: list[ClipEstimate]
) -> list[str]:
    """Formats one scenario's estimates as an aligned table.

    Args:
        true_position_xy_m: Ground-truth position in metres.
        estimates: The per-clip estimates.

    Returns:
        Lines ready to print.
    """
    lines = [
        f"  source truth = ({true_position_xy_m[0]:.1f}, "
        f"{true_position_xy_m[1]:.1f}) m",
        "  clip     x_hat     y_hat    error         D     G_hat"
        "    chi2_nu  ellipse  flag",
    ]
    for estimate in estimates:
        flag = "SINGULAR" if estimate.is_near_singular else ""
        lines.append(
            f"  {estimate.clip_index:>4}  {estimate.position_xy_m[0]:>8.2f}  "
            f"{estimate.position_xy_m[1]:>8.2f}  {estimate.error_m:>7.2f}  "
            f"{estimate.path_difference_m:>8.2f}  "
            f"{estimate.geometric_level_difference_db:>8.3f}  "
            f"{estimate.reduced_chi_square:>9.1f}  "
            f"{estimate.ellipse_semi_major_m:>7.2f}  {flag}"
        )
    errors_m = np.array([estimate.error_m for estimate in estimates])
    lines.append(
        f"  -> mean error {errors_m.mean():.2f} m | "
        f"median {np.median(errors_m):.2f} m | "
        f"max {errors_m.max():.2f} m"
    )
    return lines


def build_scenario_record(
    true_position_xy_m: Float64Array, estimates: list[ClipEstimate]
) -> dict[str, object]:
    """Summarises one scenario for the metrics document.

    Args:
        true_position_xy_m: Ground-truth position in metres.
        estimates: The per-clip estimates.

    Returns:
        A JSON-serializable record for this scenario.
    """
    errors_m = np.array([estimate.error_m for estimate in estimates])
    return {
        "true_position_xy_m": true_position_xy_m,
        "mean_error_m": float(errors_m.mean()),
        "median_error_m": float(np.median(errors_m)),
        "maximum_error_m": float(errors_m.max()),
        "clip_estimates": [
            {
                "clip_index": estimate.clip_index,
                "position_xy_m": estimate.position_xy_m,
                "error_m": estimate.error_m,
                "path_difference_m": estimate.path_difference_m,
                "geometric_level_difference_db": estimate.geometric_level_difference_db,
                "reduced_chi_square": estimate.reduced_chi_square,
                "ellipse_semi_major_m": estimate.ellipse_semi_major_m,
                "ellipse_semi_minor_m": estimate.ellipse_semi_minor_m,
                "is_near_singular": estimate.is_near_singular,
            }
            for estimate in estimates
        ],
    }


def build_metrics_document(
    configuration: SimulationConfiguration,
    sample_rate_hz: int,
    clip_count: int,
    scenarios: list[dict[str, object]],
) -> dict[str, object]:
    """Builds the whole metrics document, settings included.

    Args:
        configuration: The whole run configuration.
        sample_rate_hz: Sample rate in hertz.
        clip_count: Number of clips the recording yielded.
        scenarios: One record per source scenario.

    Returns:
        A JSON-serializable document describing the run and its results.
    """
    return {
        "configuration_directory": str(configuration.configuration_directory),
        "recording": str(configuration.data.recording.path),
        "sample_rate_hz": sample_rate_hz,
        "clip_duration_s": configuration.data.segmentation.clip_duration_s,
        "clip_count": clip_count,
        "receiver_positions_xy_m": configuration.geometry.receiver_positions_xy_m,
        "domain_size_xy_m": [
            configuration.geometry.domain.size_x_m,
            configuration.geometry.domain.size_y_m,
        ],
        "air_temperature_celsius": configuration.atmosphere.air_temperature_celsius,
        "relative_humidity_percent": configuration.atmosphere.relative_humidity_percent,
        "pressure_kpa": configuration.atmosphere.pressure_kpa,
        "signal_to_noise_ratio_db": (
            configuration.forward_model.receiver_noise.requested_signal_to_noise_ratio_db
        ),
        "random_seed": configuration.forward_model.random_seed,
        "band_center_frequencies_hz": list(
            configuration.localization.bands.center_frequencies_hz
        ),
        "window_duration_s": configuration.localization.window.duration_s,
        "window_overlap_fraction": configuration.localization.window.overlap_fraction,
        "scenarios": scenarios,
    }


def print_header(
    configuration: SimulationConfiguration,
    sample_rate_hz: int,
    sample_count: int,
    clip_count: int,
) -> None:
    """Prints the settings a run was made with.

    Args:
        configuration: The whole run configuration.
        sample_rate_hz: Sample rate in hertz.
        sample_count: Number of samples in the recording.
        clip_count: Number of clips the recording yielded.
    """
    noise = configuration.forward_model.receiver_noise
    noise_description = (
        f"white, {noise.signal_to_noise_ratio_db} dB SNR" if noise.enabled else "none"
    )
    print(f"configs     : {configuration.configuration_directory}")
    print(f"recording   : {configuration.data.recording.path}")
    print(f"sample rate : {sample_rate_hz} Hz")
    print(f"duration    : {sample_count / sample_rate_hz:.1f} s")
    print(
        f"clips       : {clip_count} x "
        f"{configuration.data.segmentation.clip_duration_s:.1f} s"
    )
    print(f"receivers   : {configuration.geometry.receiver_positions_xy_m.tolist()} m")
    print(
        f"atmosphere  : {configuration.atmosphere.air_temperature_celsius} C, "
        f"{configuration.atmosphere.relative_humidity_percent} % RH, "
        f"{configuration.atmosphere.pressure_kpa} kPa"
    )
    print(f"noise       : {noise_description}")
    print()


def main() -> None:
    """Runs every configured source scenario and writes the metrics document."""
    arguments = parse_arguments()
    configuration = load_simulation_configuration(arguments.configs)

    samples, sample_rate_hz = read_audio_file(
        configuration.data.recording.path, configuration.data.recording.as_mono
    )
    clips = segment_recording(
        samples,
        sample_rate_hz,
        configuration.data.segmentation.clip_duration_s,
        configuration.data.segmentation.overlap_fraction,
    )
    estimator = SingleSourceEstimator(
        receiver_1_xy_m=configuration.geometry.receiver_1_xy_m,
        receiver_2_xy_m=configuration.geometry.receiver_2_xy_m,
        conditions=configuration.atmosphere,
        configuration=configuration.localization,
    )
    print_header(configuration, sample_rate_hz, samples.size, int(clips.shape[0]))

    scenarios: list[dict[str, object]] = []
    for true_position_xy_m in configuration.geometry.true_source_positions_xy_m:
        estimates = estimate_over_clips(
            clips, sample_rate_hz, true_position_xy_m, estimator, configuration
        )
        for line in format_scenario_table(true_position_xy_m, estimates):
            print(line)
        print()
        scenarios.append(build_scenario_record(true_position_xy_m, estimates))

    metrics_path = write_metrics_document(
        build_metrics_document(
            configuration, sample_rate_hz, int(clips.shape[0]), scenarios
        ),
        configuration.data.output.metrics_path,
    )
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
