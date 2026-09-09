import numpy as np
import pytest

from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_equations import (
    MAXIMUM_FLAME_TILT_ANGLE_RAD,
    compute_absorption_coefficient,
    compute_base_rate_of_spread_m_per_s,
    compute_flame_tilt_angle_rad,
    compute_ignition_energy_j_per_kg,
    compute_optical_depth_m,
    compute_packing_ratio,
    compute_radiant_fraction_velocity_m_per_s,
    compute_radiative_coefficient,
    compute_rate_of_spread_balbi_2009,
    compute_reduced_rate_of_spread_balbi_2009,
    compute_upward_gas_velocity_m_per_s,
)

UPWARD_GAS_VELOCITY_M_PER_S = 2.0
ZERO = np.zeros(1)


def tilt(wind_m_per_s: float, slope_rad: float = 0.0) -> float:
    return float(
        compute_flame_tilt_angle_rad(
            np.array([wind_m_per_s]), np.array([slope_rad]), UPWARD_GAS_VELOCITY_M_PER_S
        )[0]
    )


def test_no_wind_no_slope_gives_a_vertical_flame() -> None:
    assert tilt(0.0, 0.0) == 0.0


def test_wind_alone_tilts_the_flame_forward() -> None:
    assert tilt(3.0) > 0.0


def test_tilt_is_monotone_in_wind_speed() -> None:
    wind_speeds = np.linspace(0.0, 10.0, 40)
    angles = compute_flame_tilt_angle_rad(
        wind_speeds, np.zeros_like(wind_speeds), UPWARD_GAS_VELOCITY_M_PER_S
    )
    assert np.all(np.diff(angles) > 0.0)


def test_wind_against_the_front_gives_a_backing_fire() -> None:
    assert tilt(-3.0) < 0.0


def test_uphill_slope_tilts_the_flame_forward_without_wind() -> None:
    assert tilt(0.0, np.deg2rad(20.0)) > 0.0


def test_tilt_matches_the_hand_computed_arctangent() -> None:
    expected = np.arctan(np.tan(np.deg2rad(10.0)) + 3.0 / UPWARD_GAS_VELOCITY_M_PER_S)
    assert tilt(3.0, np.deg2rad(10.0)) == np.float64(expected)


def test_tilt_is_clamped_below_the_model_breakdown_angle() -> None:
    assert tilt(1000.0) == MAXIMUM_FLAME_TILT_ANGLE_RAD
    assert tilt(-1000.0) == -MAXIMUM_FLAME_TILT_ANGLE_RAD


def test_packing_ratio_matches_table_2() -> None:
    fuel = FuelProperties.pine_needle_litter()
    assert compute_packing_ratio(fuel) == pytest.approx(0.5 / (680.0 * 0.04))
    assert compute_packing_ratio(fuel) == pytest.approx(0.02, abs=0.002)


def test_optical_depth_matches_table_2() -> None:
    fuel = FuelProperties.pine_needle_litter()
    assert compute_optical_depth_m(fuel) == pytest.approx(0.048, abs=0.005)


def test_absorption_coefficient_matches_table_2() -> None:
    fuel = FuelProperties.pine_needle_litter()
    assert compute_absorption_coefficient(fuel) == pytest.approx(0.83, abs=0.01)


def test_absorption_coefficient_is_capped_at_one() -> None:
    heavily_loaded = FuelProperties(
        fuel_density_kg_per_m3=680.0,
        moisture_content_fraction=0.10,
        surface_area_to_volume_ratio_per_m=4550.0,
        fuel_load_kg_per_m2=2.0,
        residence_time_s=20.0,
        fuel_bed_depth_m=0.04,
    )
    assert compute_absorption_coefficient(heavily_loaded) == 1.0


def test_absorption_coefficient_does_not_depend_on_bed_depth() -> None:
    shallow = FuelProperties.pine_needle_litter()
    deep = FuelProperties(
        fuel_density_kg_per_m3=680.0,
        moisture_content_fraction=0.10,
        surface_area_to_volume_ratio_per_m=4550.0,
        fuel_load_kg_per_m2=0.5,
        residence_time_s=20.0,
        fuel_bed_depth_m=2.0,
    )
    assert compute_absorption_coefficient(shallow) == pytest.approx(
        compute_absorption_coefficient(deep)
    )


def test_upward_gas_velocity_matches_table_2() -> None:
    fuel = FuelProperties.pine_needle_litter()
    assert compute_upward_gas_velocity_m_per_s(fuel) == pytest.approx(2.0)


def test_ignition_energy_is_physically_plausible() -> None:
    fuel = FuelProperties.pine_needle_litter()
    energy_j_per_kg = compute_ignition_energy_j_per_kg(fuel)
    assert energy_j_per_kg == pytest.approx(2000.0 * 300.0 + 0.10 * 2.3e6)
    assert 0.3e6 < energy_j_per_kg < 1.5e6


def test_ignition_energy_rises_with_moisture() -> None:
    dry = compute_ignition_energy_j_per_kg(FuelProperties.pine_needle_litter(0.0))
    wet = compute_ignition_energy_j_per_kg(FuelProperties.pine_needle_litter(0.30))
    assert wet > dry


def test_base_rate_of_spread_matches_table_2_case_1() -> None:
    fuel = FuelProperties.pine_needle_litter(0.10)
    assert compute_base_rate_of_spread_m_per_s(fuel) == pytest.approx(0.0027, abs=1e-4)


def test_base_rate_of_spread_matches_table_2_case_2() -> None:
    fuel = FuelProperties.pine_needle_litter(0.18)
    assert compute_base_rate_of_spread_m_per_s(fuel) == pytest.approx(0.0021, abs=1e-4)


def test_radiative_coefficient_matches_table_2_case_1() -> None:
    fuel = FuelProperties.pine_needle_litter(0.10)
    assert compute_radiative_coefficient(fuel) == pytest.approx(1.25, abs=0.01)


def test_radiative_coefficient_matches_table_2_case_2() -> None:
    fuel = FuelProperties.pine_needle_litter(0.18)
    assert compute_radiative_coefficient(fuel) == pytest.approx(0.98, abs=0.02)


def test_radiant_fraction_velocity_is_twelve_times_base_rate() -> None:
    fuel = FuelProperties.pine_needle_litter()
    assert compute_radiant_fraction_velocity_m_per_s(fuel) == pytest.approx(
        12.0 * compute_base_rate_of_spread_m_per_s(fuel)
    )
    assert compute_radiant_fraction_velocity_m_per_s(fuel) == pytest.approx(
        0.032, abs=1e-3
    )


def solve_reduced_rate_of_spread_implicitly(
    flame_tilt_angle_rad: float, radiative_coefficient: float
) -> float:
    cos_gamma = np.cos(flame_tilt_angle_rad)
    sin_gamma = np.sin(flame_tilt_angle_rad)
    gain = radiative_coefficient * (1.0 + sin_gamma - cos_gamma)
    reduced = 1.0
    for _ in range(500):
        reduced = 1.0 + gain * reduced / (1.0 + reduced / (12.0 * cos_gamma))
    return reduced


def test_equation_11_runs_hotter_than_the_implicit_equation_13b() -> None:
    fuel = FuelProperties.pine_needle_litter()
    radiative_coefficient = compute_radiative_coefficient(fuel)
    angles_rad = np.deg2rad(np.linspace(5.0, 80.0, 30))
    closed_form = compute_reduced_rate_of_spread_balbi_2009(angles_rad, fuel)
    implicit = np.array(
        [
            solve_reduced_rate_of_spread_implicitly(angle, radiative_coefficient)
            for angle in angles_rad
        ]
    )
    assert np.all(closed_form > implicit)


def test_reduced_rate_of_spread_rises_monotonically_with_tilt() -> None:
    fuel = FuelProperties.pine_needle_litter()
    angles_rad = np.deg2rad(np.linspace(1.0, 80.0, 200))
    reduced = compute_reduced_rate_of_spread_balbi_2009(angles_rad, fuel)
    assert np.all(np.diff(reduced) > 0.0)
    assert reduced[0] == pytest.approx(1.0, abs=0.2)


def test_reduced_rate_of_spread_at_the_figure_6_angles() -> None:
    fuel = FuelProperties.pine_needle_litter()
    reduced = compute_reduced_rate_of_spread_balbi_2009(
        np.deg2rad(np.array([50.0, 65.0])), fuel
    )
    assert reduced[0] == pytest.approx(10.4, abs=0.5)
    assert reduced[1] == pytest.approx(26.5, abs=0.5)


def test_zero_wind_and_zero_slope_gives_the_base_rate_of_spread() -> None:
    fuel = FuelProperties.pine_needle_litter()
    rate = compute_rate_of_spread_balbi_2009(np.zeros(1), np.zeros(1), fuel)
    assert rate[0] == pytest.approx(compute_base_rate_of_spread_m_per_s(fuel))


def test_backing_fire_creeps_at_the_base_rate_of_spread() -> None:
    fuel = FuelProperties.pine_needle_litter()
    rate = compute_rate_of_spread_balbi_2009(np.array([-1.0, -5.0]), np.zeros(2), fuel)
    np.testing.assert_allclose(rate, compute_base_rate_of_spread_m_per_s(fuel))


def test_closed_form_is_continuous_across_the_backing_threshold() -> None:
    fuel = FuelProperties.pine_needle_litter()
    base = compute_base_rate_of_spread_m_per_s(fuel)
    just_heading = compute_reduced_rate_of_spread_balbi_2009(np.array([1e-9]), fuel)
    assert just_heading[0] == pytest.approx(1.0, abs=1e-6)
    assert base > 0.0


def test_figure_3_rate_of_spread_rises_monotonically_over_field_wind_speeds() -> None:
    fuel = FuelProperties.pine_needle_litter()
    wind_speeds_m_per_s = np.linspace(0.0, 3.0, 60)
    rates = compute_rate_of_spread_balbi_2009(
        wind_speeds_m_per_s, np.zeros_like(wind_speeds_m_per_s), fuel
    )
    assert np.all(np.diff(rates) > 0.0)


def test_figure_3_starts_at_the_base_rate_and_reaches_centimetres_per_second() -> None:
    fuel = FuelProperties.pine_needle_litter()
    base = compute_base_rate_of_spread_m_per_s(fuel)
    rates = compute_rate_of_spread_balbi_2009(np.array([0.0, 3.5]), np.zeros(2), fuel)
    assert rates[0] == pytest.approx(base)
    assert 0.04 < rates[1] < 0.07
    assert rates[1] / base == pytest.approx(19.7, abs=0.5)


def test_rate_of_spread_is_monotone_in_wind_up_to_the_tilt_clamp() -> None:
    fuel = FuelProperties.pine_needle_litter()
    clamp_wind_m_per_s = np.tan(
        MAXIMUM_FLAME_TILT_ANGLE_RAD
    ) * compute_upward_gas_velocity_m_per_s(fuel)
    wind_speeds_m_per_s = np.linspace(0.0, clamp_wind_m_per_s, 300)
    rates = compute_rate_of_spread_balbi_2009(
        wind_speeds_m_per_s, np.zeros_like(wind_speeds_m_per_s), fuel
    )
    assert np.all(np.diff(rates) > 0.0)


def test_rate_of_spread_saturates_once_the_tilt_clamp_binds() -> None:
    fuel = FuelProperties.pine_needle_litter()
    clamp_wind_m_per_s = np.tan(
        MAXIMUM_FLAME_TILT_ANGLE_RAD
    ) * compute_upward_gas_velocity_m_per_s(fuel)
    rates = compute_rate_of_spread_balbi_2009(
        np.array([clamp_wind_m_per_s, 4.0 * clamp_wind_m_per_s]), np.zeros(2), fuel
    )
    assert rates[0] == pytest.approx(rates[1])
    assert clamp_wind_m_per_s == pytest.approx(11.6, abs=0.1)


def test_wetter_fuel_spreads_more_slowly() -> None:
    dry = compute_rate_of_spread_balbi_2009(
        np.array([2.0]), np.zeros(1), FuelProperties.pine_needle_litter(0.10)
    )
    wet = compute_rate_of_spread_balbi_2009(
        np.array([2.0]), np.zeros(1), FuelProperties.pine_needle_litter(0.30)
    )
    assert wet[0] < dry[0]


def test_uphill_slope_raises_the_rate_of_spread() -> None:
    fuel = FuelProperties.pine_needle_litter()
    flat = compute_rate_of_spread_balbi_2009(np.array([1.0]), np.zeros(1), fuel)
    uphill = compute_rate_of_spread_balbi_2009(
        np.array([1.0]), np.array([np.deg2rad(20.0)]), fuel
    )
    assert uphill[0] > flat[0]


def test_rate_of_spread_preserves_input_shape() -> None:
    fuel = FuelProperties.pine_needle_litter()
    wind = np.zeros((7, 8))
    assert compute_rate_of_spread_balbi_2009(wind, np.zeros_like(wind), fuel).shape == (
        7,
        8,
    )
