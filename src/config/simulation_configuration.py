import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np

from src.config.acoustic_rendering_configuration import AcousticRenderingConfiguration
from src.config.experiment_configuration import ExperimentConfiguration
from src.config.fire_simulation_configuration import FireSimulationConfiguration
from src.config.mesh_configuration import MeshConfiguration
from src.config.receiver_configuration import ReceiverConfiguration
from src.config.wind_configuration import WindConfiguration
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.utils.array_types import Float64Array

DEFAULT_CONFIGURATION_DIRECTORY: Path = Path("configs")
ENVIRONMENT_FILENAME: str = "environment.toml"
DATA_FILENAME: str = "data.toml"
GEOMETRY_FILENAME: str = "geometry.toml"
FORWARD_MODEL_FILENAME: str = "forward_model.toml"
LOCALIZATION_FILENAME: str = "localization.toml"
MESH_FILENAME: str = "mesh.toml"
WIND_FILENAME: str = "wind.toml"
FIRE_FILENAME: str = "fire.toml"
RECEIVER_FILENAME: str = "receiver.toml"
ACOUSTIC_RENDERING_FILENAME: str = "acoustic_rendering.toml"
EXPERIMENT_FILENAME: str = "experiment.toml"


@dataclass(frozen=True)
class BandConfiguration:
    """Octave band filterbank settings.

    Attributes:
        center_frequencies_hz: ISO band centres in hertz.
        filter_order: Butterworth order per band edge.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge
            may reach.
    """

    center_frequencies_hz: tuple[float, ...]
    filter_order: int
    maximum_edge_fraction_of_nyquist: float


@dataclass(frozen=True)
class WindowConfiguration:
    """Analysis window settings for the per-band level estimate.

    Attributes:
        duration_s: Window length in seconds.
        overlap_fraction: Fraction of a window shared with the previous one.
        variance_floor_db2: Floor added to each per-band variance, in dB squared.
        independent_window_fraction: Share of overlapping windows counted as
            independent when forming the variance of the mean.
    """

    duration_s: float
    overlap_fraction: float
    variance_floor_db2: float
    independent_window_fraction: float


@dataclass(frozen=True)
class DelayEstimationConfiguration:
    """Settings for the time-difference-of-arrival estimate.

    Attributes:
        interpolation_factor: Spectral zero-padding factor for sub-sample
            resolution.
        coherence_segment_sample_count: Welch segment length, in samples.
        minimum_analysis_frequency_hz: Lower edge of the coherence band, in hertz.
        maximum_analysis_frequency_fraction_of_nyquist: Upper edge of that band,
            as a fraction of Nyquist.
    """

    interpolation_factor: int
    coherence_segment_sample_count: int
    minimum_analysis_frequency_hz: float
    maximum_analysis_frequency_fraction_of_nyquist: float


@dataclass(frozen=True)
class TriangulationConfiguration:
    """Settings for inverting the two observables into a position.

    Attributes:
        domain_side_sign: Which side of the receiver baseline the domain lies on,
            resolving the mirror ambiguity.
        near_singular_tolerance: Smallest trusted distance of the level ratio from
            unity, below which the estimate is flagged.
        jacobian_step: Central finite-difference step for the covariance Jacobian.
    """

    domain_side_sign: float
    near_singular_tolerance: float
    jacobian_step: float


@dataclass(frozen=True)
class LocalizationConfiguration:
    """Everything the inverse estimator can be told.

    Attributes:
        bands: Octave band filterbank settings.
        window: Analysis window settings.
        delay_estimation: Time-difference-of-arrival settings.
        triangulation: Inversion and guard settings.
    """

    bands: BandConfiguration
    window: WindowConfiguration
    delay_estimation: DelayEstimationConfiguration
    triangulation: TriangulationConfiguration


@dataclass(frozen=True)
class RecordingConfiguration:
    """Which recording to read and how.

    Attributes:
        path: Path to the recording, relative to the repository root.
        as_mono: Whether to average the channels down to one.
    """

    path: Path
    as_mono: bool


@dataclass(frozen=True)
class SegmentationConfiguration:
    """How to cut a recording into clips.

    Attributes:
        clip_duration_s: Clip length in seconds.
        overlap_fraction: Fraction of a clip shared with the previous one.
    """

    clip_duration_s: float
    overlap_fraction: float


@dataclass(frozen=True)
class OutputConfiguration:
    """Where a run writes its metrics.

    Attributes:
        metrics_directory: Directory for metric files.
        metrics_filename: File name within that directory.
    """

    metrics_directory: Path
    metrics_filename: str

    @property
    def metrics_path(self) -> Path:
        """Full path to the metrics file."""
        return self.metrics_directory / self.metrics_filename


@dataclass(frozen=True)
class DataConfiguration:
    """Recording, segmentation and output settings.

    Attributes:
        recording: Which recording to read.
        segmentation: How to cut it into clips.
        output: Where the metrics go.
    """

    recording: RecordingConfiguration
    segmentation: SegmentationConfiguration
    output: OutputConfiguration


@dataclass(frozen=True)
class DomainConfiguration:
    """Extent of the simulated domain.

    Attributes:
        size_x_m: Extent along x in metres.
        size_y_m: Extent along y in metres.
    """

    size_x_m: float
    size_y_m: float


@dataclass(frozen=True)
class GeometryConfiguration:
    """Domain extent, receiver placement and the source scenarios to sweep.

    Attributes:
        domain: Extent of the simulated domain.
        receiver_positions_xy_m: Receiver positions, shape (2, 2).
        true_source_positions_xy_m: Ground-truth source positions, shape (n, 2).
            Used only to report error, never fed to the estimator.
    """

    domain: DomainConfiguration
    receiver_positions_xy_m: Float64Array
    true_source_positions_xy_m: Float64Array

    @property
    def receiver_1_xy_m(self) -> Float64Array:
        """Position of the first receiver."""
        return np.asarray(self.receiver_positions_xy_m[0], dtype=np.float64)

    @property
    def receiver_2_xy_m(self) -> Float64Array:
        """Position of the second receiver."""
        return np.asarray(self.receiver_positions_xy_m[1], dtype=np.float64)


@dataclass(frozen=True)
class PropagationConfiguration:
    """Free-field propagation settings.

    Attributes:
        reference_distance_m: Distance at which spreading gain is unity, in metres.
    """

    reference_distance_m: float


@dataclass(frozen=True)
class ReceiverNoiseConfiguration:
    """Additive sensor noise settings.

    Attributes:
        enabled: Whether to add noise at all.
        signal_to_noise_ratio_db: Requested ratio when noise is enabled.
    """

    enabled: bool
    signal_to_noise_ratio_db: float

    @property
    def requested_signal_to_noise_ratio_db(self) -> float | None:
        """Requested ratio, or None when noise is disabled."""
        return self.signal_to_noise_ratio_db if self.enabled else None


@dataclass(frozen=True)
class ForwardModelConfiguration:
    """Everything the forward render can be told.

    Attributes:
        propagation: Free-field propagation settings.
        receiver_noise: Additive sensor noise settings.
        random_seed: Seed making a noisy run reproducible.
    """

    propagation: PropagationConfiguration
    receiver_noise: ReceiverNoiseConfiguration
    random_seed: int


@dataclass(frozen=True)
class SimulationConfiguration:
    """One whole run, loaded from a configuration directory.

    Attributes:
        atmosphere: Air temperature, relative humidity and pressure.
        data: Recording, segmentation and output settings.
        geometry: Domain, receivers and source scenarios.
        forward_model: Propagation, noise and seed.
        localization: Inverse estimator settings.
        configuration_directory: Directory the settings were read from.
    """

    atmosphere: AtmosphericConditions
    data: DataConfiguration
    geometry: GeometryConfiguration
    forward_model: ForwardModelConfiguration
    localization: LocalizationConfiguration
    configuration_directory: Path = field(default=DEFAULT_CONFIGURATION_DIRECTORY)


def read_toml_document(path: Path) -> dict[str, Any]:
    """Reads one TOML configuration file.

    Args:
        path: Path to the file.

    Returns:
        The parsed document.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("rb") as stream:
        return tomllib.load(stream)


def to_position_array(positions: list[list[float]]) -> Float64Array:
    """Converts a list of x-y pairs into an array.

    Args:
        positions: Pairs of coordinates in metres.

    Returns:
        Array of shape (n, 2).

    Raises:
        ValueError: If the input is not a list of pairs.
    """
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("positions must be a list of [x, y] pairs in metres")
    return array


def load_atmospheric_conditions(path: Path) -> AtmosphericConditions:
    """Loads air conditions from environment.toml.

    Args:
        path: Path to the file.

    Returns:
        The air temperature, relative humidity and pressure.
    """
    section = read_toml_document(path)["atmosphere"]
    return AtmosphericConditions(
        air_temperature_celsius=float(section["air_temperature_celsius"]),
        relative_humidity_percent=float(section["relative_humidity_percent"]),
        pressure_kpa=float(section["pressure_kpa"]),
    )


def load_data_configuration(path: Path) -> DataConfiguration:
    """Loads recording, segmentation and output settings from data.toml.

    Args:
        path: Path to the file.

    Returns:
        The data settings.
    """
    document = read_toml_document(path)
    recording = document["recording"]
    segmentation = document["segmentation"]
    output = document["output"]
    return DataConfiguration(
        recording=RecordingConfiguration(
            path=Path(recording["path"]), as_mono=bool(recording["as_mono"])
        ),
        segmentation=SegmentationConfiguration(
            clip_duration_s=float(segmentation["clip_duration_s"]),
            overlap_fraction=float(segmentation["overlap_fraction"]),
        ),
        output=OutputConfiguration(
            metrics_directory=Path(output["metrics_directory"]),
            metrics_filename=str(output["metrics_filename"]),
        ),
    )


def load_geometry_configuration(path: Path) -> GeometryConfiguration:
    """Loads domain, receivers and source scenarios from geometry.toml.

    Args:
        path: Path to the file.

    Returns:
        The geometry settings.

    Raises:
        ValueError: If the file does not describe exactly two receivers.
    """
    document = read_toml_document(path)
    domain = document["domain"]
    receiver_positions_xy_m = to_position_array(document["receivers"]["positions_xy_m"])
    if receiver_positions_xy_m.shape[0] != 2:
        raise ValueError("single-source localization requires exactly two receivers")
    return GeometryConfiguration(
        domain=DomainConfiguration(
            size_x_m=float(domain["size_x_m"]), size_y_m=float(domain["size_y_m"])
        ),
        receiver_positions_xy_m=receiver_positions_xy_m,
        true_source_positions_xy_m=to_position_array(
            document["sources"]["true_positions_xy_m"]
        ),
    )


def load_forward_model_configuration(path: Path) -> ForwardModelConfiguration:
    """Loads propagation, noise and seed from forward_model.toml.

    Args:
        path: Path to the file.

    Returns:
        The forward model settings.
    """
    document = read_toml_document(path)
    propagation = document["propagation"]
    receiver_noise = document["receiver_noise"]
    return ForwardModelConfiguration(
        propagation=PropagationConfiguration(
            reference_distance_m=float(propagation["reference_distance_m"])
        ),
        receiver_noise=ReceiverNoiseConfiguration(
            enabled=bool(receiver_noise["enabled"]),
            signal_to_noise_ratio_db=float(receiver_noise["signal_to_noise_ratio_db"]),
        ),
        random_seed=int(document["random"]["seed"]),
    )


def load_localization_configuration(path: Path) -> LocalizationConfiguration:
    """Loads estimator settings from localization.toml.

    Args:
        path: Path to the file.

    Returns:
        The localization settings.
    """
    document = read_toml_document(path)
    bands = document["bands"]
    window = document["window"]
    delay_estimation = document["delay_estimation"]
    triangulation = document["triangulation"]
    return LocalizationConfiguration(
        bands=BandConfiguration(
            center_frequencies_hz=tuple(
                float(value) for value in bands["center_frequencies_hz"]
            ),
            filter_order=int(bands["filter_order"]),
            maximum_edge_fraction_of_nyquist=float(
                bands["maximum_edge_fraction_of_nyquist"]
            ),
        ),
        window=WindowConfiguration(
            duration_s=float(window["duration_s"]),
            overlap_fraction=float(window["overlap_fraction"]),
            variance_floor_db2=float(window["variance_floor_db2"]),
            independent_window_fraction=float(window["independent_window_fraction"]),
        ),
        delay_estimation=DelayEstimationConfiguration(
            interpolation_factor=int(delay_estimation["interpolation_factor"]),
            coherence_segment_sample_count=int(
                delay_estimation["coherence_segment_sample_count"]
            ),
            minimum_analysis_frequency_hz=float(
                delay_estimation["minimum_analysis_frequency_hz"]
            ),
            maximum_analysis_frequency_fraction_of_nyquist=float(
                delay_estimation["maximum_analysis_frequency_fraction_of_nyquist"]
            ),
        ),
        triangulation=TriangulationConfiguration(
            domain_side_sign=float(triangulation["domain_side_sign"]),
            near_singular_tolerance=float(triangulation["near_singular_tolerance"]),
            jacobian_step=float(triangulation["jacobian_step"]),
        ),
    )


def load_simulation_configuration(
    configuration_directory: Path = DEFAULT_CONFIGURATION_DIRECTORY,
) -> SimulationConfiguration:
    """Loads every configuration file in one directory.

    Args:
        configuration_directory: Directory holding the five TOML files.

    Returns:
        The whole run configuration.
    """
    return SimulationConfiguration(
        atmosphere=load_atmospheric_conditions(
            configuration_directory / ENVIRONMENT_FILENAME
        ),
        data=load_data_configuration(configuration_directory / DATA_FILENAME),
        geometry=load_geometry_configuration(
            configuration_directory / GEOMETRY_FILENAME
        ),
        forward_model=load_forward_model_configuration(
            configuration_directory / FORWARD_MODEL_FILENAME
        ),
        localization=load_localization_configuration(
            configuration_directory / LOCALIZATION_FILENAME
        ),
        configuration_directory=configuration_directory,
    )


@dataclass(frozen=True)
class ForwardSimulationConfiguration:
    """One whole forward render, loaded from a configuration directory.

    Sits alongside `SimulationConfiguration` rather than inside it: the two
    pipelines share only the air conditions, and the inverse estimator must
    never be able to reach the fire model through its own configuration.

    Attributes:
        experiment: What this scene is called, so its outputs are self-describing.
        mesh: Extent, resolution and connectivity of the grid.
        wind: Speed and direction of the constant wind field.
        fire: Fuel, ignition point, run length and emission model.
        receiver: Receiver layout strategy and its parameters.
        acoustic: Sampling and signal parameters of the render.
        atmosphere: Air temperature, relative humidity and pressure.
        configuration_directory: Directory the settings were read from.
    """

    mesh: MeshConfiguration
    wind: WindConfiguration
    fire: FireSimulationConfiguration
    receiver: ReceiverConfiguration
    acoustic: AcousticRenderingConfiguration
    atmosphere: AtmosphericConditions
    experiment: ExperimentConfiguration = field(default_factory=ExperimentConfiguration)
    configuration_directory: Path = field(default=DEFAULT_CONFIGURATION_DIRECTORY)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild from the mapping `to_dict` produced.

        A saved run is therefore self-describing: the figure exporter and the
        application can replay the exact scene a run came from without being
        told which configuration directory produced it.

        Args:
            data: Mapping of section name to that section's fields.

        Returns:
            The configuration, with the directory left at its default.
        """
        atmosphere = data.get("atmosphere", {})
        return cls(
            experiment=ExperimentConfiguration.from_dict(data.get("experiment", {})),
            mesh=MeshConfiguration.from_dict(data.get("mesh", {})),
            wind=WindConfiguration.from_dict(data.get("wind", {})),
            fire=FireSimulationConfiguration.from_dict(data.get("fire", {})),
            receiver=ReceiverConfiguration.from_dict(data.get("receiver", {})),
            acoustic=AcousticRenderingConfiguration.from_dict(
                data.get("acoustic_rendering", {})
            ),
            atmosphere=AtmosphericConditions(
                air_temperature_celsius=float(
                    atmosphere.get("air_temperature_celsius", 15.0)
                ),
                relative_humidity_percent=float(
                    atmosphere.get("relative_humidity_percent", 70.0)
                ),
                pressure_kpa=float(atmosphere.get("pressure_kpa", 101.325)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render every section as one JSON-serialisable mapping.

        Returns:
            Mapping of section name to that section's fields.
        """
        return {
            "experiment": self.experiment.to_dict(),
            "mesh": self.mesh.to_dict(),
            "wind": self.wind.to_dict(),
            "fire": self.fire.to_dict(),
            "receiver": self.receiver.to_dict(),
            "acoustic_rendering": self.acoustic.to_dict(),
            "atmosphere": {
                "air_temperature_celsius": self.atmosphere.air_temperature_celsius,
                "relative_humidity_percent": self.atmosphere.relative_humidity_percent,
                "pressure_kpa": self.atmosphere.pressure_kpa,
            },
        }


def load_experiment_configuration(path: Path) -> ExperimentConfiguration:
    """Loads the scene's identity from experiment.toml.

    A directory without one is still a valid scene; it just falls back to the
    default name, so an older configuration directory keeps working.

    Args:
        path: Path to the file.

    Returns:
        The experiment identity.
    """
    if not path.is_file():
        return ExperimentConfiguration()
    return ExperimentConfiguration.from_dict(read_toml_document(path)["experiment"])


def load_forward_simulation_configuration(
    configuration_directory: Path = DEFAULT_CONFIGURATION_DIRECTORY,
) -> ForwardSimulationConfiguration:
    """Loads every forward-render configuration file in one directory.

    Args:
        configuration_directory: Directory holding the six TOML files.

    Returns:
        The whole forward run configuration.
    """
    return ForwardSimulationConfiguration(
        experiment=load_experiment_configuration(
            configuration_directory / EXPERIMENT_FILENAME
        ),
        mesh=MeshConfiguration.from_dict(
            read_toml_document(configuration_directory / MESH_FILENAME)["mesh"]
        ),
        wind=WindConfiguration.from_dict(
            read_toml_document(configuration_directory / WIND_FILENAME)["wind"]
        ),
        fire=FireSimulationConfiguration.from_dict(
            read_toml_document(configuration_directory / FIRE_FILENAME)["fire"]
        ),
        receiver=ReceiverConfiguration.from_dict(
            read_toml_document(configuration_directory / RECEIVER_FILENAME)["receiver"]
        ),
        acoustic=AcousticRenderingConfiguration.from_dict(
            read_toml_document(configuration_directory / ACOUSTIC_RENDERING_FILENAME)[
                "acoustic_rendering"
            ]
        ),
        atmosphere=load_atmospheric_conditions(
            configuration_directory / ENVIRONMENT_FILENAME
        ),
        configuration_directory=configuration_directory,
    )
