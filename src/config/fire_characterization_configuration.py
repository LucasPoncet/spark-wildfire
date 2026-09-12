"""Settings for characterising a fire from a sequence of imaging maps.

Sits alongside the localisation configuration rather than inside it. The two
read the same correlation curves but form different maps from them, and the
detection settings must stay exactly as they are or every committed
localisation result stops being reproducible.

The imaging map is described as a set of overrides on the scene's own
correlation and search settings, not as a second full copy of them. A scene
that changes its whitening band changes it for both families, which is what
keeps the point spread function a property of the array rather than of one
configuration block drifting away from another.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.config.multi_source_localization_configuration import (
    CorrelationConfiguration,
    SteeredResponsePowerConfiguration,
)
from src.config.simulation_configuration import read_toml_document

FIRE_CHARACTERIZATION_FILENAME: str = "fire_characterization.toml"


@dataclass(frozen=True)
class ImagingMapConfiguration:
    """How the imaging map differs from the detection map.

    The plan writes the pooling as "volumetric". That is not a value the map
    code takes: its rules are `sum`, `mean`, `max` and `point`, of which the
    first three all pool over the cell's delay interval and only `point`
    samples the centre. "Volumetric" therefore names the family, and this field
    names the member of it, settled by the Tier 0 audit rather than assumed.

    Attributes:
        phase_transform_exponent: The `beta` the imaging family uses.
        pairwise_combinator: How pairwise maps merge, `sum` for an additive map.
        pooling: Which interval pooling rule, `sum`, `mean` or `max`.
        grid_spacing_m: Imaging cell side length, in metres.
        frame_duration_s: Length of one frame, in seconds.
        frame_hop_s: Advance between consecutive frames, in seconds.
        background_percentile: Percentile of a frame treated as background
            when a first moment is being taken.
        extent_background_percentile: The same, for a second moment. The two
            need not agree and on this scene must not: a centroid is dragged by
            the tails the domain boundary truncates, so it wants them gone,
            while a second moment *is* the tails and shrinks without them. What
            the extent stage actually requires is only that the observed
            covariance and the response covariance be prepared the same way as
            each other, which holds for any percentile as long as one is used
            for both sides.
    """

    phase_transform_exponent: float
    pairwise_combinator: str
    pooling: str
    grid_spacing_m: float
    frame_duration_s: float
    frame_hop_s: float
    background_percentile: float
    extent_background_percentile: float

    def apply_to_correlation(
        self, correlation: CorrelationConfiguration
    ) -> CorrelationConfiguration:
        """Derives the imaging correlation settings from the scene's own.

        Args:
            correlation: The scene's detection correlation settings.

        Returns:
            The same settings with the imaging exponent and framing.
        """
        return replace(
            correlation,
            phase_transform_exponent=self.phase_transform_exponent,
        )

    def apply_to_steered_response_power(
        self, steered_response_power: SteeredResponsePowerConfiguration
    ) -> SteeredResponsePowerConfiguration:
        """Derives the imaging map settings from the scene's own.

        Args:
            steered_response_power: The scene's detection search settings.

        Returns:
            The same settings with the imaging combinator, pooling and grid.
        """
        return replace(
            steered_response_power,
            pooling=self.pooling,
            pairwise_combinator=self.pairwise_combinator,
            coarse_grid_spacing_m=self.grid_spacing_m,
        )


@dataclass(frozen=True)
class PointSpreadFunctionConfiguration:
    """Where the array response is tabulated and where that table is kept.

    Attributes:
        calibration_grid_spacing_m: Node spacing of the calibration field.
        centroid_bias_field_path: Measured centroid offsets.
        covariance_field_path: Analytic response covariances.
    """

    calibration_grid_spacing_m: float
    centroid_bias_field_path: Path
    covariance_field_path: Path


@dataclass(frozen=True)
class BearingConfiguration:
    """Settings for the three bearing estimators and their fusion.

    Attributes:
        direction_count: Directions swept when maximising the skewness.
        minimum_centroid_displacement_m: Below this total drift the direction
            is noise and the drift estimator refuses rather than guessing.
        bootstrap_resample_count: Resamples over frames for the interval.
        fusion_weights: Weight per method name in the circular fusion.
    """

    direction_count: int
    minimum_centroid_displacement_m: float
    bootstrap_resample_count: int
    fusion_weights: dict[str, float]


@dataclass(frozen=True)
class ExtentConfiguration:
    """Settings turning a second moment into a front semi-axis.

    Attributes:
        shape_factor: Multiplier from the square root of an eigenvalue to a
            semi-axis. The square root of two is the thin circular ring.
        shape_factor_path: Where the calibrated factor is written.
        minimum_resolved_frames: Fewest resolved frames a rate of spread may
            be fitted from.
    """

    shape_factor: float
    shape_factor_path: Path
    minimum_resolved_frames: int


@dataclass(frozen=True)
class SpreadModelConfiguration:
    """Settings for the elliptical spread model fit.

    Attributes:
        maximum_iterations: Iteration ceiling for the least-squares fit.
        parameter_tolerance: Convergence tolerance on the parameters.
        seed_count: How many independent seeds the fit is repeated from.
    """

    maximum_iterations: int
    parameter_tolerance: float
    seed_count: int


@dataclass(frozen=True)
class FrontConfiguration:
    """Settings for the front position stage.

    Attributes:
        radial_profile_step_m: Sampling step along a radial profile.
        maximum_radius_m: Furthest radius a profile is extracted to.
        deconvolution_iteration_count: Richardson-Lucy iterations.
        total_variation_weight: Regularisation weight on the source density.
        contour_level_fraction: Floor level, as a fraction of the density's
            global peak, a contour is extracted at.
        contour_local_level_fraction: Level as a fraction of each ray's own
            peak, which a direction must clear as well as the floor. It is
            what lets a front be found where the fire burns dimly; zero leaves
            the global floor as the only level.
        contour_support_fraction: Least wedge mass a direction must hold, as a
            fraction of the best direction's, before the contour will claim a
            front there. A level is measured against the density's global peak
            and so says nothing about the direction it is read in; this does,
            and it is what keeps an open arc open.
    """

    radial_profile_step_m: float
    maximum_radius_m: float
    deconvolution_iteration_count: int
    total_variation_weight: float
    contour_level_fraction: float
    contour_local_level_fraction: float
    contour_support_fraction: float


@dataclass(frozen=True)
class FireCharacterizationConfiguration:
    """Every setting the characterisation tiers read.

    Attributes:
        imaging_map: How the imaging map differs from the detection map.
        point_spread_function: Response calibration settings.
        bearing: Bearing estimator settings.
        extent: Extent and shape factor settings.
        spread_model: Elliptical model fit settings.
        front: Front position settings.
    """

    imaging_map: ImagingMapConfiguration
    point_spread_function: PointSpreadFunctionConfiguration
    bearing: BearingConfiguration
    extent: ExtentConfiguration
    spread_model: SpreadModelConfiguration
    front: FrontConfiguration


def load_fire_characterization_configuration(
    path: Path,
) -> FireCharacterizationConfiguration:
    """Reads `fire_characterization.toml`.

    Args:
        path: Path to the file.

    Returns:
        The settings.
    """
    document: dict[str, Any] = read_toml_document(path)
    imaging = document["imaging_map"]
    response = document["point_spread_function"]
    bearing = document["bearing"]
    extent = document["extent"]
    spread_model = document["spread_model"]
    front = document["front"]
    return FireCharacterizationConfiguration(
        imaging_map=ImagingMapConfiguration(
            phase_transform_exponent=float(imaging["phase_transform_exponent"]),
            pairwise_combinator=str(imaging["pairwise_combinator"]),
            pooling=str(imaging["pooling"]),
            grid_spacing_m=float(imaging["grid_spacing_m"]),
            frame_duration_s=float(imaging["frame_duration_s"]),
            frame_hop_s=float(imaging["frame_hop_s"]),
            background_percentile=float(imaging["background_percentile"]),
            extent_background_percentile=float(imaging["extent_background_percentile"]),
        ),
        point_spread_function=PointSpreadFunctionConfiguration(
            calibration_grid_spacing_m=float(response["calibration_grid_spacing_m"]),
            centroid_bias_field_path=Path(str(response["centroid_bias_field_path"])),
            covariance_field_path=Path(str(response["covariance_field_path"])),
        ),
        bearing=BearingConfiguration(
            direction_count=int(bearing["direction_count"]),
            minimum_centroid_displacement_m=float(
                bearing["minimum_centroid_displacement_m"]
            ),
            bootstrap_resample_count=int(bearing["bootstrap_resample_count"]),
            fusion_weights={
                str(name): float(weight)
                for name, weight in bearing["fusion_weights"].items()
            },
        ),
        extent=ExtentConfiguration(
            shape_factor=float(extent["shape_factor"]),
            shape_factor_path=Path(str(extent["shape_factor_path"])),
            minimum_resolved_frames=int(extent["minimum_resolved_frames"]),
        ),
        spread_model=SpreadModelConfiguration(
            maximum_iterations=int(spread_model["maximum_iterations"]),
            parameter_tolerance=float(spread_model["parameter_tolerance"]),
            seed_count=int(spread_model["seed_count"]),
        ),
        front=FrontConfiguration(
            radial_profile_step_m=float(front["radial_profile_step_m"]),
            maximum_radius_m=float(front["maximum_radius_m"]),
            deconvolution_iteration_count=int(front["deconvolution_iteration_count"]),
            total_variation_weight=float(front["total_variation_weight"]),
            contour_level_fraction=float(front["contour_level_fraction"]),
            contour_local_level_fraction=float(front["contour_local_level_fraction"]),
            contour_support_fraction=float(front["contour_support_fraction"]),
        ),
    )
