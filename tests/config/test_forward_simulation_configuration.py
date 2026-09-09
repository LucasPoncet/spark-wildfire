import json
from pathlib import Path

import pytest

from src.config.acoustic_rendering_configuration import (
    AcousticRenderingConfiguration,
    compute_observation_stride,
)
from src.config.fire_simulation_configuration import FireSimulationConfiguration
from src.config.mesh_configuration import MeshConfiguration
from src.config.receiver_configuration import ReceiverConfiguration
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    ForwardSimulationConfiguration,
    load_forward_simulation_configuration,
)
from src.config.wind_configuration import WindConfiguration
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions

SECTION_NAMES = (
    "experiment",
    "mesh",
    "wind",
    "fire",
    "receiver",
    "acoustic_rendering",
    "atmosphere",
)


def load_default() -> ForwardSimulationConfiguration:
    return load_forward_simulation_configuration(DEFAULT_CONFIGURATION_DIRECTORY)


def test_the_shipped_configuration_directory_loads() -> None:
    config = load_default()
    assert config.configuration_directory == DEFAULT_CONFIGURATION_DIRECTORY
    assert config.mesh.extent_x_m > 0.0
    assert config.fire.simulation_duration_s > 0.0


def test_every_section_is_serialised() -> None:
    document = load_default().to_dict()
    assert sorted(document) == sorted(SECTION_NAMES)


def test_the_serialised_configuration_is_json_writable(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(load_default().to_dict(), indent=2), encoding="utf-8")
    assert sorted(json.loads(path.read_text(encoding="utf-8"))) == sorted(SECTION_NAMES)


def test_a_missing_configuration_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        load_forward_simulation_configuration(tmp_path / "absent")


def test_the_shipped_ignition_point_is_off_the_ring_centre() -> None:
    config = load_default()
    assert (
        config.fire.ignition_cell_x_fraction,
        config.fire.ignition_cell_y_fraction,
    ) != (
        config.receiver.ring_center_x_fraction,
        config.receiver.ring_center_y_fraction,
    )


def test_the_shipped_observation_interval_is_not_shorter_than_a_fire_step() -> None:
    config = load_default()
    assert compute_observation_stride(config.acoustic.observation_interval_s, 18.0) >= 1


def test_an_observation_interval_shorter_than_the_fire_step_raises() -> None:
    config = load_default()
    with pytest.raises(ValueError, match="shorter than the fire timestep"):
        compute_observation_stride(
            config.acoustic.observation_interval_s,
            config.acoustic.observation_interval_s * 2.0,
        )


def test_a_serialised_configuration_rebuilds_itself() -> None:
    original = load_default()
    assert ForwardSimulationConfiguration.from_dict(original.to_dict()).to_dict() == (
        original.to_dict()
    )


def test_a_saved_run_config_rebuilds_the_scene_it_came_from() -> None:
    original = load_forward_simulation_configuration(Path("configs/e2"))
    rebuilt = ForwardSimulationConfiguration.from_dict(
        json.loads(json.dumps(original.to_dict()))
    )
    assert rebuilt.fire.spread_engine_name == original.fire.spread_engine_name
    assert (
        rebuilt.fire.ignition_points_xy_fraction
        == original.fire.ignition_points_xy_fraction
    )
    assert rebuilt.mesh.extent_x_m == original.mesh.extent_x_m
    assert rebuilt.atmosphere == original.atmosphere


def test_an_empty_mapping_rebuilds_the_defaults() -> None:
    assert ForwardSimulationConfiguration.from_dict({}).to_dict() == (
        ForwardSimulationConfiguration(
            mesh=MeshConfiguration(),
            wind=WindConfiguration(),
            fire=FireSimulationConfiguration(),
            receiver=ReceiverConfiguration(),
            acoustic=AcousticRenderingConfiguration(),
            atmosphere=AtmosphericConditions(15.0, 70.0, 101.325),
        ).to_dict()
    )
