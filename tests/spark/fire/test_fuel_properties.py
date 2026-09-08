import dataclasses

import pytest

from src.spark.fire.fuel_properties import FuelProperties


def test_preset_matches_balbi_2009_table_1() -> None:
    fuel = FuelProperties.pine_needle_litter()
    assert fuel.fuel_density_kg_per_m3 == 680.0
    assert fuel.moisture_content_fraction == 0.10
    assert fuel.surface_area_to_volume_ratio_per_m == 4550.0
    assert fuel.fuel_load_kg_per_m2 == 0.5
    assert fuel.residence_time_s == 20.0
    assert fuel.fuel_bed_depth_m == 0.04


def test_preset_accepts_the_second_tabulated_moisture() -> None:
    assert FuelProperties.pine_needle_litter(0.18).moisture_content_fraction == 0.18


def test_fields_are_frozen() -> None:
    fuel = FuelProperties.pine_needle_litter()
    with pytest.raises(dataclasses.FrozenInstanceError):
        fuel.fuel_load_kg_per_m2 = 1.0  # type: ignore[misc]


def test_instantiating_directly_keeps_every_field() -> None:
    fuel = FuelProperties(
        fuel_density_kg_per_m3=500.0,
        moisture_content_fraction=0.2,
        surface_area_to_volume_ratio_per_m=6000.0,
        fuel_load_kg_per_m2=0.8,
        residence_time_s=15.0,
        fuel_bed_depth_m=0.05,
    )
    assert dataclasses.asdict(fuel)["fuel_bed_depth_m"] == 0.05
