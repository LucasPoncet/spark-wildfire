from pathlib import Path

import pytest

from src.config.experiment_configuration import (
    DEFAULT_EXPERIMENT_NAME,
    ExperimentConfiguration,
)
from src.config.simulation_configuration import load_forward_simulation_configuration

LADDER_NAMES = ("e1", "e2", "e3", "e4")


def test_the_default_name_is_a_safe_directory_name() -> None:
    assert ExperimentConfiguration().name == DEFAULT_EXPERIMENT_NAME


def test_a_configuration_round_trips() -> None:
    original = ExperimentConfiguration(name="e7", title="E7", description="a scene")
    assert ExperimentConfiguration.from_dict(original.to_dict()) == original


def test_an_empty_mapping_gives_the_defaults() -> None:
    assert ExperimentConfiguration.from_dict({}) == ExperimentConfiguration()


@pytest.mark.parametrize(
    "unsafe_name", ["../escape", "with space", "Upper", "", "a/b", "e1;rm"]
)
def test_an_unsafe_name_is_rejected(unsafe_name: str) -> None:
    with pytest.raises(ValueError, match="must be lower case"):
        ExperimentConfiguration.from_dict({"name": unsafe_name})


@pytest.mark.parametrize("safe_name", ["e1", "e10", "front_only", "two-sources", "9x"])
def test_a_safe_name_is_accepted(safe_name: str) -> None:
    assert ExperimentConfiguration.from_dict({"name": safe_name}).name == safe_name


@pytest.mark.parametrize("experiment_name", LADDER_NAMES)
def test_every_ladder_directory_names_itself(experiment_name: str) -> None:
    config = load_forward_simulation_configuration(Path("configs") / experiment_name)
    assert config.experiment.name == experiment_name
    assert config.experiment.title != ""
    assert config.experiment.description != ""


def test_the_shipped_default_directory_names_itself() -> None:
    config = load_forward_simulation_configuration(Path("configs"))
    assert config.experiment.name == DEFAULT_EXPERIMENT_NAME


def test_a_directory_without_an_experiment_file_still_loads(tmp_path: Path) -> None:
    for filename in (
        "environment.toml",
        "mesh.toml",
        "wind.toml",
        "fire.toml",
        "receiver.toml",
        "acoustic_rendering.toml",
    ):
        (tmp_path / filename).write_text(
            (Path("configs") / filename).read_text(encoding="utf-8"), encoding="utf-8"
        )
    config = load_forward_simulation_configuration(tmp_path)
    assert config.experiment.name == DEFAULT_EXPERIMENT_NAME


def test_the_experiment_section_survives_serialisation() -> None:
    config = load_forward_simulation_configuration(Path("configs/e3"))
    assert config.to_dict()["experiment"]["name"] == "e3"
