from pathlib import Path

import pytest

from src.config.fire_characterization_configuration import (
    FIRE_CHARACTERIZATION_FILENAME,
    FireCharacterizationConfiguration,
    load_fire_characterization_configuration,
)
from src.config.multi_source_localization_configuration import (
    MAXIMUM_POOLING,
    MEAN_POOLING,
    POINT_POOLING,
    SUM_POOLING,
)
from src.config.simulation_configuration import (
    SimulationConfiguration,
    load_simulation_configuration,
)

SCENE_DIRECTORY: Path = Path("configs/f1")
VOLUMETRIC_POOLINGS: tuple[str, ...] = (SUM_POOLING, MEAN_POOLING, MAXIMUM_POOLING)


@pytest.fixture(scope="module")
def characterization() -> FireCharacterizationConfiguration:
    return load_fire_characterization_configuration(
        SCENE_DIRECTORY / FIRE_CHARACTERIZATION_FILENAME
    )


@pytest.fixture(scope="module")
def scene() -> SimulationConfiguration:
    return load_simulation_configuration(SCENE_DIRECTORY)


def test_the_scene_ships_a_characterization_file() -> None:
    assert (SCENE_DIRECTORY / FIRE_CHARACTERIZATION_FILENAME).is_file()


def test_every_section_loads(
    characterization: FireCharacterizationConfiguration,
) -> None:
    assert characterization.imaging_map.grid_spacing_m > 0.0
    assert characterization.point_spread_function.calibration_grid_spacing_m > 0.0
    assert characterization.bearing.direction_count > 0
    assert characterization.extent.minimum_resolved_frames > 0
    assert characterization.spread_model.seed_count > 0
    assert characterization.front.contour_level_fraction > 0.0


def test_the_imaging_pooling_names_a_rule_the_map_code_accepts(
    characterization: FireCharacterizationConfiguration,
) -> None:
    assert characterization.imaging_map.pooling in VOLUMETRIC_POOLINGS


def test_the_imaging_pooling_is_volumetric_rather_than_point_sampled(
    characterization: FireCharacterizationConfiguration,
) -> None:
    assert characterization.imaging_map.pooling != POINT_POOLING


def test_the_frame_hop_does_not_exceed_the_frame_duration(
    characterization: FireCharacterizationConfiguration,
) -> None:
    assert (
        characterization.imaging_map.frame_hop_s
        <= characterization.imaging_map.frame_duration_s
    )


def test_the_background_percentile_is_a_percentile(
    characterization: FireCharacterizationConfiguration,
) -> None:
    assert 0.0 <= characterization.imaging_map.background_percentile <= 100.0


def test_the_imaging_grid_is_finer_than_the_detection_search_grid(
    characterization: FireCharacterizationConfiguration, scene: SimulationConfiguration
) -> None:
    detection = scene.localization.multi_source.steered_response_power
    assert characterization.imaging_map.grid_spacing_m < detection.coarse_grid_spacing_m


def test_the_imaging_overrides_leave_the_detection_band_alone(
    characterization: FireCharacterizationConfiguration, scene: SimulationConfiguration
) -> None:
    detection = scene.localization.multi_source.correlation
    imaging = characterization.imaging_map.apply_to_correlation(detection)
    assert imaging.lowest_frequency_hz == detection.lowest_frequency_hz
    assert imaging.highest_frequency_hz == detection.highest_frequency_hz
    assert imaging.use_analytic_envelope == detection.use_analytic_envelope


def test_the_imaging_overrides_replace_the_exponent(
    characterization: FireCharacterizationConfiguration, scene: SimulationConfiguration
) -> None:
    detection = scene.localization.multi_source.correlation
    imaging = characterization.imaging_map.apply_to_correlation(detection)
    assert (
        imaging.phase_transform_exponent
        == characterization.imaging_map.phase_transform_exponent
    )


def test_the_imaging_overrides_replace_the_combinator_and_the_grid(
    characterization: FireCharacterizationConfiguration, scene: SimulationConfiguration
) -> None:
    detection = scene.localization.multi_source.steered_response_power
    imaging = characterization.imaging_map.apply_to_steered_response_power(detection)
    assert (
        imaging.pairwise_combinator == characterization.imaging_map.pairwise_combinator
    )
    assert imaging.pooling == characterization.imaging_map.pooling
    assert imaging.coarse_grid_spacing_m == characterization.imaging_map.grid_spacing_m
    assert imaging.candidate_height_m == detection.candidate_height_m


def test_a_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        load_fire_characterization_configuration(tmp_path / "absent.toml")
