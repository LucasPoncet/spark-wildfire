from pathlib import Path

import numpy as np
import pytest

from config.simulation_configuration import (
    DATA_FILENAME,
    ENVIRONMENT_FILENAME,
    FORWARD_MODEL_FILENAME,
    GEOMETRY_FILENAME,
    LOCALIZATION_FILENAME,
    SimulationConfiguration,
    load_geometry_configuration,
    load_simulation_configuration,
    read_toml_document,
    to_position_array,
)
from tests.conftest import CONFIGURATION_DIRECTORY

EXPECTED_FILENAMES = (
    ENVIRONMENT_FILENAME,
    DATA_FILENAME,
    GEOMETRY_FILENAME,
    FORWARD_MODEL_FILENAME,
    LOCALIZATION_FILENAME,
)


@pytest.mark.parametrize("filename", EXPECTED_FILENAMES)
def test_every_configuration_file_is_present(filename: str) -> None:
    assert (CONFIGURATION_DIRECTORY / filename).is_file()


def test_missing_configuration_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_toml_document(tmp_path / "absent.toml")


def test_configuration_directory_loads_end_to_end(
    simulation_configuration: SimulationConfiguration,
) -> None:
    assert isinstance(simulation_configuration, SimulationConfiguration)
    assert simulation_configuration.configuration_directory == CONFIGURATION_DIRECTORY


def test_atmosphere_is_physically_plausible(
    simulation_configuration: SimulationConfiguration,
) -> None:
    atmosphere = simulation_configuration.atmosphere
    assert -50.0 < atmosphere.air_temperature_celsius < 60.0
    assert 0.0 <= atmosphere.relative_humidity_percent <= 100.0
    assert atmosphere.pressure_kpa > 0.0


def test_receiver_positions_are_a_pair_of_distinct_points(
    simulation_configuration: SimulationConfiguration,
) -> None:
    positions = simulation_configuration.geometry.receiver_positions_xy_m
    assert positions.shape == (2, 2)
    assert not np.allclose(positions[0], positions[1])


def test_source_positions_lie_inside_the_domain(
    simulation_configuration: SimulationConfiguration,
) -> None:
    geometry = simulation_configuration.geometry
    sources = geometry.true_source_positions_xy_m
    assert np.all(sources[:, 0] >= 0.0)
    assert np.all(sources[:, 0] <= geometry.domain.size_x_m)
    assert np.all(sources[:, 1] >= 0.0)
    assert np.all(sources[:, 1] <= geometry.domain.size_y_m)


def test_band_centres_are_ascending_octaves(
    simulation_configuration: SimulationConfiguration,
) -> None:
    centers = np.asarray(
        simulation_configuration.localization.bands.center_frequencies_hz
    )
    assert centers.size > 1
    assert np.all(np.diff(centers) > 0.0)
    assert np.allclose(centers[1:] / centers[:-1], 2.0)


def test_window_configuration_is_within_valid_ranges(
    simulation_configuration: SimulationConfiguration,
) -> None:
    window = simulation_configuration.localization.window
    assert window.duration_s > 0.0
    assert 0.0 <= window.overlap_fraction < 1.0
    assert window.variance_floor_db2 > 0.0
    assert 0.0 < window.independent_window_fraction <= 1.0


def test_recording_path_is_relative_to_the_repository(
    simulation_configuration: SimulationConfiguration,
) -> None:
    assert not simulation_configuration.data.recording.path.is_absolute()


def test_metrics_path_joins_directory_and_filename(
    simulation_configuration: SimulationConfiguration,
) -> None:
    output = simulation_configuration.data.output
    assert output.metrics_path == output.metrics_directory / output.metrics_filename


def test_disabled_noise_reports_no_signal_to_noise_ratio(
    simulation_configuration: SimulationConfiguration,
) -> None:
    noise = simulation_configuration.forward_model.receiver_noise
    if noise.enabled:
        assert (
            noise.requested_signal_to_noise_ratio_db == noise.signal_to_noise_ratio_db
        )
    else:
        assert noise.requested_signal_to_noise_ratio_db is None


def test_position_array_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="list of"):
        to_position_array([[1.0, 2.0, 3.0]])


def test_geometry_requires_exactly_two_receivers(tmp_path: Path) -> None:
    path = tmp_path / GEOMETRY_FILENAME
    path.write_text(
        "[domain]\nsize_x_m = 100.0\nsize_y_m = 100.0\n\n"
        "[receivers]\npositions_xy_m = [[0.0, 0.0]]\n\n"
        "[sources]\ntrue_positions_xy_m = [[50.0, 50.0]]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly two receivers"):
        load_geometry_configuration(path)


def test_an_edited_configuration_directory_changes_the_loaded_values(
    tmp_path: Path,
) -> None:
    for filename in EXPECTED_FILENAMES:
        (tmp_path / filename).write_text(
            (CONFIGURATION_DIRECTORY / filename).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (tmp_path / ENVIRONMENT_FILENAME).write_text(
        "[atmosphere]\nair_temperature_celsius = 30.0\n"
        "relative_humidity_percent = 40.0\npressure_kpa = 95.0\n",
        encoding="utf-8",
    )
    configuration = load_simulation_configuration(tmp_path)
    assert configuration.atmosphere.air_temperature_celsius == 30.0
    assert configuration.atmosphere.relative_humidity_percent == 40.0
    assert configuration.atmosphere.pressure_kpa == 95.0
