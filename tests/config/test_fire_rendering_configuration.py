import json
from pathlib import Path

import pytest

from src.config.fire_rendering_configuration import FireRenderingConfiguration


def test_default_config_roundtrips_through_json(tmp_path: Path) -> None:
    original = FireRenderingConfiguration.default()
    config_path = tmp_path / "config.json"
    original.to_json(config_path)
    restored = FireRenderingConfiguration.from_json(config_path)

    assert restored == original
    assert restored.mesh == original.mesh
    assert restored.wind == original.wind
    assert restored.fire == original.fire
    assert restored.receiver == original.receiver
    assert restored.atmosphere == original.atmosphere
    assert restored.acoustic == original.acoustic


def test_written_json_is_indented_and_readable(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    FireRenderingConfiguration.default().to_json(config_path)
    text = config_path.read_text(encoding="utf-8")

    assert "\n  " in text
    assert set(json.loads(text)) == {
        "mesh",
        "wind",
        "fire",
        "receiver",
        "atmosphere",
        "acoustic",
    }


def test_partial_override_preserves_defaults() -> None:
    config = FireRenderingConfiguration.from_dict({"mesh": {"extent_x_m": 50.0}})

    assert config.mesh.extent_x_m == 50.0
    assert config.mesh.cell_spacing_m == 0.5
    assert config.wind.wind_speed_m_per_s == 2.0


def test_empty_dict_gives_all_defaults() -> None:
    assert (
        FireRenderingConfiguration.from_dict({}) == FireRenderingConfiguration.default()
    )


def test_to_dict_round_trips_through_from_dict() -> None:
    original = FireRenderingConfiguration.from_dict(
        {
            "mesh": {"extent_x_m": 42.0, "use_diagonal_neighbors": False},
            "receiver": {"placement_strategy": "grid", "grid_spacing_m": 5.0},
        }
    )
    assert FireRenderingConfiguration.from_dict(original.to_dict()) == original


def test_pressure_is_converted_to_the_kilopascals_the_model_expects() -> None:
    atmosphere = FireRenderingConfiguration.default().atmosphere
    assert atmosphere.atmospheric_pressure_pa == 101325.0
    assert atmosphere.atmospheric_pressure_kpa == 101.325


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"absent\.json"):
        FireRenderingConfiguration.from_json(tmp_path / "absent.json")
