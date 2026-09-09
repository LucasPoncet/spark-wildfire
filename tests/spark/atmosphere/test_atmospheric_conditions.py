from dataclasses import replace

import pytest

from src.spark.atmosphere.atmospheric_conditions import (
    REFERENCE_PRESSURE_KPA,
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)


def test_speed_of_sound_at_fifteen_celsius() -> None:
    assert compute_speed_of_sound_m_per_s(15.0) == pytest.approx(340.3, abs=0.5)


def test_speed_of_sound_at_zero_celsius() -> None:
    assert compute_speed_of_sound_m_per_s(0.0) == pytest.approx(331.3, abs=0.5)


def test_speed_of_sound_rises_with_temperature() -> None:
    assert compute_speed_of_sound_m_per_s(30.0) > compute_speed_of_sound_m_per_s(10.0)


def test_conditions_default_to_standard_pressure() -> None:
    assert AtmosphericConditions(15.0, 70.0).pressure_kpa == REFERENCE_PRESSURE_KPA


def test_replacing_a_condition_leaves_the_original_unchanged() -> None:
    conditions = AtmosphericConditions(15.0, 70.0)
    warmer = replace(conditions, air_temperature_celsius=25.0)
    assert conditions.air_temperature_celsius == 15.0
    assert warmer.air_temperature_celsius == 25.0


def test_equal_conditions_compare_equal() -> None:
    assert AtmosphericConditions(15.0, 70.0) == AtmosphericConditions(15.0, 70.0)
