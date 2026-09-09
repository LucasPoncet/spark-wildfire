import numpy as np
import pytest
from matplotlib.figure import Figure

from src.spark.fire.fuel_properties import FuelProperties
from src.utils.visualization.rate_of_spread_plotter import (
    compute_rate_of_spread_curve_mm_per_s,
    plot_rate_of_spread_against_moisture,
    plot_rate_of_spread_against_wind_speed,
    plot_rate_of_spread_panels,
)

WIND_SPEEDS_M_PER_S = np.linspace(0.0, 10.0, 50)
SLOPE_ANGLES_RAD = np.radians(np.array([0.0, 20.0]))
MOISTURE_FRACTIONS = np.linspace(0.05, 0.25, 20)
TABLE_2_BASE_RATE_MM_PER_S = 2.67


def build_fuel() -> FuelProperties:
    return FuelProperties.pine_needle_litter()


def test_the_curve_starts_at_the_published_base_rate() -> None:
    curve_mm_per_s = compute_rate_of_spread_curve_mm_per_s(
        WIND_SPEEDS_M_PER_S, 0.0, build_fuel()
    )
    assert curve_mm_per_s[0] == pytest.approx(TABLE_2_BASE_RATE_MM_PER_S, rel=0.02)


def test_the_curve_rises_with_wind_speed() -> None:
    curve_mm_per_s = compute_rate_of_spread_curve_mm_per_s(
        WIND_SPEEDS_M_PER_S, 0.0, build_fuel()
    )
    assert bool(np.all(np.diff(curve_mm_per_s) >= -1e-9))
    assert curve_mm_per_s[-1] > curve_mm_per_s[0]


def test_an_uphill_slope_lifts_the_whole_curve() -> None:
    fuel = build_fuel()
    flat_mm_per_s = compute_rate_of_spread_curve_mm_per_s(
        WIND_SPEEDS_M_PER_S, 0.0, fuel
    )
    uphill_mm_per_s = compute_rate_of_spread_curve_mm_per_s(
        WIND_SPEEDS_M_PER_S, float(np.radians(20.0)), fuel
    )
    assert bool(np.all(uphill_mm_per_s >= flat_mm_per_s - 1e-9))


def test_the_wind_figure_has_one_axes_and_a_curve_per_slope() -> None:
    figure = plot_rate_of_spread_against_wind_speed(
        WIND_SPEEDS_M_PER_S, SLOPE_ANGLES_RAD, build_fuel()
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].get_lines()) == SLOPE_ANGLES_RAD.size


def test_the_moisture_figure_has_one_axes_and_a_curve_per_wind_speed() -> None:
    wind_speeds_m_per_s = np.array([0.0, 3.0])
    figure = plot_rate_of_spread_against_moisture(
        MOISTURE_FRACTIONS, wind_speeds_m_per_s
    )
    assert len(figure.axes) == 1
    assert len(figure.axes[0].get_lines()) == wind_speeds_m_per_s.size


def test_the_panel_figure_has_two_axes() -> None:
    figure = plot_rate_of_spread_panels(
        WIND_SPEEDS_M_PER_S, SLOPE_ANGLES_RAD, MOISTURE_FRACTIONS, build_fuel()
    )
    assert len(figure.axes) == 2


def test_the_axes_carry_unit_labels() -> None:
    figure = plot_rate_of_spread_against_wind_speed(
        WIND_SPEEDS_M_PER_S, SLOPE_ANGLES_RAD, build_fuel()
    )
    axes = figure.axes[0]
    assert "m/s" in axes.get_xlabel()
    assert "mm/s" in axes.get_ylabel()
