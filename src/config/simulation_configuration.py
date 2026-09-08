import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.utils.array_types import Float64Array

DEFAULT_CONFIGURATION_DIRECTORY: Path = Path("configs")
ENVIRONMENT_FILENAME: str = "environment.toml"
DATA_FILENAME: str = "data.toml"
GEOMETRY_FILENAME: str = "geometry.toml"
FORWARD_MODEL_FILENAME: str = "forward_model.toml"
LOCALIZATION_FILENAME: str = "localization.toml"


@dataclass(frozen=True)
class BandConfiguration:
    center_frequencies_hz: tuple[float, ...]
    filter_order: int
    maximum_edge_fraction_of_nyquist: float


@dataclass(frozen=True)
class WindowConfiguration:
    duration_s: float
    overlap_fraction: float
    variance_floor_db2: float
    independent_window_fraction: float


@dataclass(frozen=True)
class DelayEstimationConfiguration:
    interpolation_factor: int
    coherence_segment_sample_count: int
    minimum_analysis_frequency_hz: float
    maximum_analysis_frequency_fraction_of_nyquist: float


@dataclass(frozen=True)
class TriangulationConfiguration:
    domain_side_sign: float
    near_singular_tolerance: float
    jacobian_step: float


@dataclass(frozen=True)
class LocalizationConfiguration:
    bands: BandConfiguration
    window: WindowConfiguration
    delay_estimation: DelayEstimationConfiguration
    triangulation: TriangulationConfiguration


@dataclass(frozen=True)
class RecordingConfiguration:
    path: Path
    as_mono: bool


@dataclass(frozen=True)
class SegmentationConfiguration:
    clip_duration_s: float
    overlap_fraction: float


@dataclass(frozen=True)
class OutputConfiguration:
    metrics_directory: Path
    metrics_filename: str

    @property
    def metrics_path(self) -> Path:
        return self.metrics_directory / self.metrics_filename


@dataclass(frozen=True)
class DataConfiguration:
    recording: RecordingConfiguration
    segmentation: SegmentationConfiguration
    output: OutputConfiguration


@dataclass(frozen=True)
class DomainConfiguration:
    size_x_m: float
    size_y_m: float


@dataclass(frozen=True)
class GeometryConfiguration:
    domain: DomainConfiguration
    receiver_positions_xy_m: Float64Array
    true_source_positions_xy_m: Float64Array

    @property
    def receiver_1_xy_m(self) -> Float64Array:
        return self.receiver_positions_xy_m[0]

    @property
    def receiver_2_xy_m(self) -> Float64Array:
        return self.receiver_positions_xy_m[1]


@dataclass(frozen=True)
class PropagationConfiguration:
    reference_distance_m: float


@dataclass(frozen=True)
class ReceiverNoiseConfiguration:
    enabled: bool
    signal_to_noise_ratio_db: float

    @property
    def requested_signal_to_noise_ratio_db(self) -> float | None:
        return self.signal_to_noise_ratio_db if self.enabled else None


@dataclass(frozen=True)
class ForwardModelConfiguration:
    propagation: PropagationConfiguration
    receiver_noise: ReceiverNoiseConfiguration
    random_seed: int


@dataclass(frozen=True)
class SimulationConfiguration:
    atmosphere: AtmosphericConditions
    data: DataConfiguration
    geometry: GeometryConfiguration
    forward_model: ForwardModelConfiguration
    localization: LocalizationConfiguration
    configuration_directory: Path = field(default=DEFAULT_CONFIGURATION_DIRECTORY)


def read_toml_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("rb") as stream:
        return tomllib.load(stream)


def to_position_array(positions: list[list[float]]) -> Float64Array:
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("positions must be a list of [x, y] pairs in metres")
    return array


def load_atmospheric_conditions(path: Path) -> AtmosphericConditions:
    section = read_toml_document(path)["atmosphere"]
    return AtmosphericConditions(
        air_temperature_celsius=float(section["air_temperature_celsius"]),
        relative_humidity_percent=float(section["relative_humidity_percent"]),
        pressure_kpa=float(section["pressure_kpa"]),
    )


def load_data_configuration(path: Path) -> DataConfiguration:
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
        true_source_positions_xy_m=to_position_array(document["sources"]["true_positions_xy_m"]),
    )


def load_forward_model_configuration(path: Path) -> ForwardModelConfiguration:
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
    document = read_toml_document(path)
    bands = document["bands"]
    window = document["window"]
    delay_estimation = document["delay_estimation"]
    triangulation = document["triangulation"]
    return LocalizationConfiguration(
        bands=BandConfiguration(
            center_frequencies_hz=tuple(float(value) for value in bands["center_frequencies_hz"]),
            filter_order=int(bands["filter_order"]),
            maximum_edge_fraction_of_nyquist=float(bands["maximum_edge_fraction_of_nyquist"]),
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
    return SimulationConfiguration(
        atmosphere=load_atmospheric_conditions(configuration_directory / ENVIRONMENT_FILENAME),
        data=load_data_configuration(configuration_directory / DATA_FILENAME),
        geometry=load_geometry_configuration(configuration_directory / GEOMETRY_FILENAME),
        forward_model=load_forward_model_configuration(
            configuration_directory / FORWARD_MODEL_FILENAME
        ),
        localization=load_localization_configuration(
            configuration_directory / LOCALIZATION_FILENAME
        ),
        configuration_directory=configuration_directory,
    )
