"""Everything the K-source, N-receiver estimator can be told."""

from dataclasses import dataclass
from typing import Any

SUM_COMBINATOR: str = "sum"
# Sums the pooled pair maps as they are, with no per-pair rescaling. That
# rescaling reads a minimum and a maximum off the mixture, so it makes the
# combined map depend on which sources are present and is the larger of the
# two things that stop an imaging map adding. Detection keeps "sum".
UNSCALED_SUM_COMBINATOR: str = "unscaled_sum"
PRODUCT_COMBINATOR: str = "product"
HARMONIC_MEAN_COMBINATOR: str = "harmonic_mean"

SUM_POOLING: str = "sum"
MEAN_POOLING: str = "mean"
MAXIMUM_POOLING: str = "max"
POINT_POOLING: str = "point"

PARABOLIC_INTERPOLATION: str = "parabolic"
EXPONENTIAL_INTERPOLATION: str = "exponential"

NOTCH_DEFLATION: str = "tdoa_notch"
SUBSPACE_PROJECTION_DEFLATION: str = "subspace_projection"


@dataclass(frozen=True)
class CorrelationConfiguration:
    """Settings for the generalised cross-correlation the whole search runs on.

    Attributes:
        lowest_frequency_hz: Lower edge of the whitening band, in hertz.
        highest_frequency_hz: Upper edge of that band, in hertz.
        phase_transform_exponent: The `beta` of the parameterised phase
            transform. One recovers conventional PHAT, zero plain
            cross-correlation.
        phase_transform_regularization: Additive term keeping the whitening
            denominator away from zero.
        use_analytic_envelope: Whether to take the magnitude of the analytic
            signal of the correlation, removing the band-pass sinc ripples that
            otherwise become spurious map peaks.
        window_duration_s: Length of one accumulation window, in seconds.
        window_overlap: Fraction of a window shared with the previous one.
        maximum_absolute_lag_margin: Multiplier on the largest geometrically
            possible lag, setting how far the curve is retained.
        peak_interpolation: Sub-sample refinement, "parabolic" or "exponential".
    """

    lowest_frequency_hz: float
    highest_frequency_hz: float
    phase_transform_exponent: float
    phase_transform_regularization: float
    use_analytic_envelope: bool
    window_duration_s: float
    window_overlap: float
    maximum_absolute_lag_margin: float
    peak_interpolation: str


@dataclass(frozen=True)
class FractionalBandConfiguration:
    """Settings for the fractional-octave filterbank the level stage runs on.

    Attributes:
        fraction_denominator: Bands per octave.
        lowest_centre_hz: Lowest band centre to consider, in hertz.
        highest_centre_hz: Highest band centre to consider, in hertz.
        codec_cutoff_margin: Fraction of the measured codec cutoff a band's
            upper edge may reach.
        minimum_band_signal_to_noise_ratio_db: Smallest margin over the noise
            floor a band needs to be used.
        filter_order: Butterworth order per band edge.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may
            reach.
        codec_detection_window_sample_count: Analysis window for the cutoff
            measurement, in samples.
        codec_detection_overlap_fraction: Overlap for that measurement.
        codec_detection_floor_drop_db: Drop below the spectral peak that counts
            as the encoder noise floor.
    """

    fraction_denominator: int
    lowest_centre_hz: float
    highest_centre_hz: float
    codec_cutoff_margin: float
    minimum_band_signal_to_noise_ratio_db: float
    filter_order: int
    maximum_edge_fraction_of_nyquist: float
    codec_detection_window_sample_count: int
    codec_detection_overlap_fraction: float
    codec_detection_floor_drop_db: float


@dataclass(frozen=True)
class SteeredResponsePowerConfiguration:
    """Settings for the steered response power map and its refinement.

    Attributes:
        pooling: How a cell's delay interval is reduced to one value: "sum",
            "mean", "max", or "point" for the cell centre only.
        pairwise_combinator: How pairwise maps are merged: "sum", "product" or
            "harmonic_mean".
        coarse_grid_spacing_m: Cell size of the first search level, in metres.
        refinement_levels: Number of coarse-to-fine levels.
        refinement_factor: Subdivision factor between levels.
        refinement_retained_fraction: Fraction of cells carried to the next
            level.
        low_frequency_seed_hz: Upper frequency used on the first level, whose
            wider main lobe stops coarse cells falling between basins.
        candidate_height_m: Height of the candidate grid above the ground plane.
    """

    pooling: str
    pairwise_combinator: str
    coarse_grid_spacing_m: float
    refinement_levels: int
    refinement_factor: int
    refinement_retained_fraction: float
    low_frequency_seed_hz: float
    candidate_height_m: float


@dataclass(frozen=True)
class DeflationConfiguration:
    """Settings for peeling one located source out of the correlations.

    Attributes:
        method: "tdoa_notch" or "subspace_projection".
        maximum_source_count: Ceiling on how many sources the loop may accept.
        peak_prominence_threshold_ratio: Smallest accepted ratio of the map peak
            over the map median.
        minimum_source_separation_m: Closest an accepted source may sit to one
            already accepted.
        residual_power_reduction_threshold: Smallest drop in residual
            correlation energy that counts as progress.
        notch_width_s: Half-width of the time-difference-of-arrival notch, in
            seconds.
    """

    method: str
    maximum_source_count: int
    peak_prominence_threshold_ratio: float
    minimum_source_separation_m: float
    residual_power_reduction_threshold: float
    notch_width_s: float


@dataclass(frozen=True)
class PositionRefinementConfiguration:
    """Settings for the Gauss-Newton refinement of a seeded position.

    Attributes:
        maximum_gauss_newton_iterations: Iteration ceiling.
        position_convergence_tolerance_m: Step size below which the solve stops.
        use_band_level_terms: Whether to add the band-level residuals to the
            delay residuals, which is what supplies the radial direction.
    """

    maximum_gauss_newton_iterations: int
    position_convergence_tolerance_m: float
    use_band_level_terms: bool


@dataclass(frozen=True)
class WindowedClusteringConfiguration:
    """Settings for the independent second route to the source count.

    Attributes:
        enabled: Whether to run the cross-check at all.
        window_duration_s: Length of one short window, in seconds.
        window_overlap: Fraction of a window shared with the previous one.
        clustering_radius_m: Distance within which per-window maxima are taken
            to belong to the same source.
        minimum_cluster_size: Smallest number of windows a cluster needs to
            count as a source.
    """

    enabled: bool
    window_duration_s: float
    window_overlap: float
    clustering_radius_m: float
    minimum_cluster_size: int


@dataclass(frozen=True)
class LocalizationMetricsConfiguration:
    """Settings for scoring a set of estimates against a set of true sources.

    Attributes:
        optimal_subpattern_assignment_cutoff_m: Distance at which a localization
            error costs as much as a missed or spurious source, in metres.
        optimal_subpattern_assignment_order: Exponent of the underlying distance.
    """

    optimal_subpattern_assignment_cutoff_m: float
    optimal_subpattern_assignment_order: float


@dataclass(frozen=True)
class MultiSourceLocalizationConfiguration:
    """Everything the K-source estimator can be told.

    Attributes:
        correlation: Generalised cross-correlation settings.
        bands: Fractional-octave filterbank settings.
        steered_response_power: Map, pooling and refinement settings.
        deflation: Source peeling settings.
        refinement: Gauss-Newton settings.
        clustering: Windowed-clustering cross-check settings.
        metrics: How a result is scored against ground truth.
    """

    correlation: CorrelationConfiguration
    bands: FractionalBandConfiguration
    steered_response_power: SteeredResponsePowerConfiguration
    deflation: DeflationConfiguration
    refinement: PositionRefinementConfiguration
    clustering: WindowedClusteringConfiguration
    metrics: LocalizationMetricsConfiguration


def load_correlation_configuration(section: dict[str, Any]) -> CorrelationConfiguration:
    """Reads the `[correlation]` section.

    Args:
        section: The parsed section.

    Returns:
        The correlation settings.
    """
    return CorrelationConfiguration(
        lowest_frequency_hz=float(section["lowest_frequency_hz"]),
        highest_frequency_hz=float(section["highest_frequency_hz"]),
        phase_transform_exponent=float(section["phase_transform_exponent"]),
        phase_transform_regularization=float(section["phase_transform_regularization"]),
        use_analytic_envelope=bool(section["use_analytic_envelope"]),
        window_duration_s=float(section["window_duration_s"]),
        window_overlap=float(section["window_overlap"]),
        maximum_absolute_lag_margin=float(section["maximum_absolute_lag_margin"]),
        peak_interpolation=str(section["peak_interpolation"]),
    )


def load_fractional_band_configuration(
    section: dict[str, Any],
) -> FractionalBandConfiguration:
    """Reads the `[fractional_bands]` section.

    Args:
        section: The parsed section.

    Returns:
        The filterbank settings.
    """
    return FractionalBandConfiguration(
        fraction_denominator=int(section["fraction_denominator"]),
        lowest_centre_hz=float(section["lowest_centre_hz"]),
        highest_centre_hz=float(section["highest_centre_hz"]),
        codec_cutoff_margin=float(section["codec_cutoff_margin"]),
        minimum_band_signal_to_noise_ratio_db=float(
            section["minimum_band_signal_to_noise_ratio_db"]
        ),
        filter_order=int(section["filter_order"]),
        maximum_edge_fraction_of_nyquist=float(
            section["maximum_edge_fraction_of_nyquist"]
        ),
        codec_detection_window_sample_count=int(
            section["codec_detection_window_sample_count"]
        ),
        codec_detection_overlap_fraction=float(
            section["codec_detection_overlap_fraction"]
        ),
        codec_detection_floor_drop_db=float(section["codec_detection_floor_drop_db"]),
    )


def load_steered_response_power_configuration(
    section: dict[str, Any],
) -> SteeredResponsePowerConfiguration:
    """Reads the `[steered_response_power]` section.

    Args:
        section: The parsed section.

    Returns:
        The map and refinement settings.
    """
    return SteeredResponsePowerConfiguration(
        pooling=str(section["pooling"]),
        pairwise_combinator=str(section["pairwise_combinator"]),
        coarse_grid_spacing_m=float(section["coarse_grid_spacing_m"]),
        refinement_levels=int(section["refinement_levels"]),
        refinement_factor=int(section["refinement_factor"]),
        refinement_retained_fraction=float(section["refinement_retained_fraction"]),
        low_frequency_seed_hz=float(section["low_frequency_seed_hz"]),
        candidate_height_m=float(section["candidate_height_m"]),
    )


def load_deflation_configuration(section: dict[str, Any]) -> DeflationConfiguration:
    """Reads the `[deflation]` section.

    Args:
        section: The parsed section.

    Returns:
        The peeling settings.
    """
    return DeflationConfiguration(
        method=str(section["method"]),
        maximum_source_count=int(section["maximum_source_count"]),
        peak_prominence_threshold_ratio=float(
            section["peak_prominence_threshold_ratio"]
        ),
        minimum_source_separation_m=float(section["minimum_source_separation_m"]),
        residual_power_reduction_threshold=float(
            section["residual_power_reduction_threshold"]
        ),
        notch_width_s=float(section["notch_width_s"]),
    )


def load_position_refinement_configuration(
    section: dict[str, Any],
) -> PositionRefinementConfiguration:
    """Reads the `[refinement]` section.

    Args:
        section: The parsed section.

    Returns:
        The Gauss-Newton settings.
    """
    return PositionRefinementConfiguration(
        maximum_gauss_newton_iterations=int(section["maximum_gauss_newton_iterations"]),
        position_convergence_tolerance_m=float(
            section["position_convergence_tolerance_m"]
        ),
        use_band_level_terms=bool(section["use_band_level_terms"]),
    )


def load_windowed_clustering_configuration(
    section: dict[str, Any],
) -> WindowedClusteringConfiguration:
    """Reads the `[windowed_clustering]` section.

    Args:
        section: The parsed section.

    Returns:
        The cross-check settings.
    """
    return WindowedClusteringConfiguration(
        enabled=bool(section["enabled"]),
        window_duration_s=float(section["window_duration_s"]),
        window_overlap=float(section["window_overlap"]),
        clustering_radius_m=float(section["clustering_radius_m"]),
        minimum_cluster_size=int(section["minimum_cluster_size"]),
    )


def load_localization_metrics_configuration(
    section: dict[str, Any],
) -> LocalizationMetricsConfiguration:
    """Reads the `[localization_metrics]` section.

    Args:
        section: The parsed section.

    Returns:
        The scoring settings.
    """
    return LocalizationMetricsConfiguration(
        optimal_subpattern_assignment_cutoff_m=float(
            section["optimal_subpattern_assignment_cutoff_m"]
        ),
        optimal_subpattern_assignment_order=float(
            section["optimal_subpattern_assignment_order"]
        ),
    )


def load_multi_source_localization_configuration(
    document: dict[str, Any],
) -> MultiSourceLocalizationConfiguration:
    """Reads every multi-source section of a localization document.

    Args:
        document: The parsed `localization.toml`.

    Returns:
        The whole multi-source estimator configuration.
    """
    return MultiSourceLocalizationConfiguration(
        correlation=load_correlation_configuration(document["correlation"]),
        bands=load_fractional_band_configuration(document["fractional_bands"]),
        steered_response_power=load_steered_response_power_configuration(
            document["steered_response_power"]
        ),
        deflation=load_deflation_configuration(document["deflation"]),
        refinement=load_position_refinement_configuration(document["refinement"]),
        clustering=load_windowed_clustering_configuration(
            document["windowed_clustering"]
        ),
        metrics=load_localization_metrics_configuration(
            document["localization_metrics"]
        ),
    )
