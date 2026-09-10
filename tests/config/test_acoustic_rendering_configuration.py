import pytest

from src.config.acoustic_rendering_configuration import (
    AcousticRenderingConfiguration,
    compute_observation_stride,
)

FIRE_TIME_STEP_S = 18.0


def test_observation_interval_has_a_default() -> None:
    assert AcousticRenderingConfiguration().observation_interval_s == 60.0


def test_observation_interval_round_trips() -> None:
    original = AcousticRenderingConfiguration(observation_interval_s=30.0)
    assert AcousticRenderingConfiguration.from_dict(original.to_dict()) == original


def test_observation_interval_survives_a_partial_mapping() -> None:
    config = AcousticRenderingConfiguration.from_dict({"observation_interval_s": 120.0})
    assert config.observation_interval_s == 120.0
    assert config.sample_rate_hz == AcousticRenderingConfiguration().sample_rate_hz


def test_non_positive_observation_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="observation interval must be positive"):
        AcousticRenderingConfiguration.from_dict({"observation_interval_s": 0.0})


def test_negative_observation_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="observation interval must be positive"):
        AcousticRenderingConfiguration.from_dict({"observation_interval_s": -1.0})


def test_stride_is_the_interval_in_whole_fire_steps() -> None:
    assert compute_observation_stride(90.0, FIRE_TIME_STEP_S) == 5


def test_stride_rounds_to_the_nearest_whole_step() -> None:
    assert compute_observation_stride(100.0, FIRE_TIME_STEP_S) == 6


def test_stride_of_one_when_the_interval_equals_the_timestep() -> None:
    assert compute_observation_stride(FIRE_TIME_STEP_S, FIRE_TIME_STEP_S) == 1


def test_interval_shorter_than_the_fire_timestep_is_rejected() -> None:
    with pytest.raises(ValueError, match="shorter than the fire timestep"):
        compute_observation_stride(5.0, FIRE_TIME_STEP_S)
