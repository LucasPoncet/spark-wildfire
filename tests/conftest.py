from pathlib import Path

import pytest

from src.config.simulation_configuration import (
    BandConfiguration,
    DelayEstimationConfiguration,
    ForwardModelConfiguration,
    GeometryConfiguration,
    LocalizationConfiguration,
    SimulationConfiguration,
    TriangulationConfiguration,
    WindowConfiguration,
    load_simulation_configuration,
)
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.utils.array_types import Float64Array

CONFIGURATION_DIRECTORY: Path = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def simulation_configuration() -> SimulationConfiguration:
    return load_simulation_configuration(CONFIGURATION_DIRECTORY)


@pytest.fixture(scope="session")
def atmosphere(simulation_configuration: SimulationConfiguration) -> AtmosphericConditions:
    return simulation_configuration.atmosphere


@pytest.fixture(scope="session")
def geometry(simulation_configuration: SimulationConfiguration) -> GeometryConfiguration:
    return simulation_configuration.geometry


@pytest.fixture(scope="session")
def forward_model(simulation_configuration: SimulationConfiguration) -> ForwardModelConfiguration:
    return simulation_configuration.forward_model


@pytest.fixture(scope="session")
def reference_distance_m(forward_model: ForwardModelConfiguration) -> float:
    return forward_model.propagation.reference_distance_m


@pytest.fixture(scope="session")
def localization(simulation_configuration: SimulationConfiguration) -> LocalizationConfiguration:
    return simulation_configuration.localization


@pytest.fixture(scope="session")
def bands(localization: LocalizationConfiguration) -> BandConfiguration:
    return localization.bands


@pytest.fixture(scope="session")
def window(localization: LocalizationConfiguration) -> WindowConfiguration:
    return localization.window


@pytest.fixture(scope="session")
def delay_estimation(localization: LocalizationConfiguration) -> DelayEstimationConfiguration:
    return localization.delay_estimation


@pytest.fixture(scope="session")
def triangulation(localization: LocalizationConfiguration) -> TriangulationConfiguration:
    return localization.triangulation


@pytest.fixture(scope="session")
def receiver_1_xy_m(geometry: GeometryConfiguration) -> Float64Array:
    return geometry.receiver_1_xy_m


@pytest.fixture(scope="session")
def receiver_2_xy_m(geometry: GeometryConfiguration) -> Float64Array:
    return geometry.receiver_2_xy_m


@pytest.fixture(scope="session")
def sample_rate_hz() -> int:
    return 44100


@pytest.fixture(scope="session")
def clip_duration_s(simulation_configuration: SimulationConfiguration) -> float:
    return simulation_configuration.data.segmentation.clip_duration_s
