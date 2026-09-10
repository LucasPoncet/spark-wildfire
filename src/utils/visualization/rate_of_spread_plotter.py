"""The Balbi 2009 rate of spread as published curves.

Takes data and returns a `matplotlib.figure.Figure`. Never calls `savefig`.
Reproduces the shape of Balbi 2009 Fig. 3 and Fig. 4 against the Table 2
Case 1 fuel the equation tests already validate, so the figure and the unit
tests agree by construction rather than by coincidence.
"""

import numpy as np
from matplotlib.figure import Figure

from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.utils.array_types import Float64Array

MILLIMETRES_PER_METRE: float = 1000.0


def compute_rate_of_spread_curve_mm_per_s(
    wind_speeds_m_per_s: Float64Array,
    slope_angle_rad: float,
    fuel: FuelProperties,
) -> Float64Array:
    """Rate of spread across a sweep of wind speeds, in millimetres per second.

    Args:
        wind_speeds_m_per_s: Wind speeds along the front normal, in m/s.
        slope_angle_rad: Terrain slope along the front normal, in radians.
        fuel: The fuel bed.

    Returns:
        Float64 array the same shape as the wind speeds.
    """
    rate_of_spread_m_per_s = compute_rate_of_spread_balbi_2009(
        wind_speeds_m_per_s,
        np.full_like(wind_speeds_m_per_s, slope_angle_rad),
        fuel,
    )
    return np.asarray(rate_of_spread_m_per_s * MILLIMETRES_PER_METRE, dtype=np.float64)


def plot_rate_of_spread_against_wind_speed(
    wind_speeds_m_per_s: Float64Array,
    slope_angles_rad: Float64Array,
    fuel: FuelProperties,
) -> Figure:
    """Rate of spread against wind speed, one curve per slope.

    Args:
        wind_speeds_m_per_s: Wind speeds along the front normal, in m/s.
        slope_angles_rad: One slope per curve, in radians.
        fuel: The fuel bed.

    Returns:
        A matplotlib Figure. The caller saves or displays it.
    """
    figure = Figure(figsize=(3.5, 2.8))
    axes = figure.add_subplot(111)
    for slope_angle_rad in slope_angles_rad:
        axes.plot(
            wind_speeds_m_per_s,
            compute_rate_of_spread_curve_mm_per_s(
                wind_speeds_m_per_s, float(slope_angle_rad), fuel
            ),
            label=f"slope {np.degrees(slope_angle_rad):.0f}°",
        )
    axes.set_xlabel("wind speed along front normal (m/s)")
    axes.set_ylabel("rate of spread (mm/s)")
    axes.set_title("Balbi 2009 Eq. 11, Table 2 Case 1")
    axes.legend(fontsize="x-small")
    axes.grid(True, linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_rate_of_spread_against_moisture(
    moisture_content_fractions: Float64Array,
    wind_speeds_m_per_s: Float64Array,
) -> Figure:
    """Rate of spread against fuel moisture, one curve per wind speed.

    Args:
        moisture_content_fractions: Moisture contents as fractions, not percent.
        wind_speeds_m_per_s: One wind speed per curve, in m/s.

    Returns:
        A matplotlib Figure. The caller saves or displays it.
    """
    figure = Figure(figsize=(3.5, 2.8))
    axes = figure.add_subplot(111)
    for wind_speed_m_per_s in wind_speeds_m_per_s:
        rates_mm_per_s = np.array(
            [
                float(
                    compute_rate_of_spread_balbi_2009(
                        np.array([wind_speed_m_per_s]),
                        np.zeros(1),
                        FuelProperties.pine_needle_litter(float(moisture_fraction)),
                    )[0]
                )
                * MILLIMETRES_PER_METRE
                for moisture_fraction in moisture_content_fractions
            ],
            dtype=np.float64,
        )
        axes.plot(
            moisture_content_fractions * 100.0,
            rates_mm_per_s,
            label=f"U = {wind_speed_m_per_s:.0f} m/s",
        )
    axes.set_xlabel("fuel moisture content (%)")
    axes.set_ylabel("rate of spread (mm/s)")
    axes.set_title("Moisture damping")
    axes.legend(fontsize="x-small")
    axes.grid(True, linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_rate_of_spread_panels(
    wind_speeds_m_per_s: Float64Array,
    slope_angles_rad: Float64Array,
    moisture_content_fractions: Float64Array,
    fuel: FuelProperties,
) -> Figure:
    """Wind, slope and moisture dependence in one two-panel figure.

    Args:
        wind_speeds_m_per_s: Wind speeds along the front normal, in m/s.
        slope_angles_rad: One slope per curve in the left panel, in radians.
        moisture_content_fractions: Moisture contents for the right panel.
        fuel: The fuel bed used in the left panel.

    Returns:
        A matplotlib Figure with two axes.
    """
    figure = Figure(figsize=(7.0, 2.8))
    wind_axes = figure.add_subplot(121)
    for slope_angle_rad in slope_angles_rad:
        wind_axes.plot(
            wind_speeds_m_per_s,
            compute_rate_of_spread_curve_mm_per_s(
                wind_speeds_m_per_s, float(slope_angle_rad), fuel
            ),
            label=f"slope {np.degrees(slope_angle_rad):.0f}°",
        )
    wind_axes.set_xlabel("wind speed along front normal (m/s)")
    wind_axes.set_ylabel("rate of spread (mm/s)")
    wind_axes.set_title("Wind and slope")
    wind_axes.legend(fontsize="x-small")
    wind_axes.grid(True, linewidth=0.3, alpha=0.5)

    moisture_axes = figure.add_subplot(122)
    for wind_speed_m_per_s in (0.0, float(np.max(wind_speeds_m_per_s)) / 2.0):
        rates_mm_per_s = np.array(
            [
                float(
                    compute_rate_of_spread_balbi_2009(
                        np.array([wind_speed_m_per_s]),
                        np.zeros(1),
                        FuelProperties.pine_needle_litter(float(moisture_fraction)),
                    )[0]
                )
                * MILLIMETRES_PER_METRE
                for moisture_fraction in moisture_content_fractions
            ],
            dtype=np.float64,
        )
        moisture_axes.plot(
            moisture_content_fractions * 100.0,
            rates_mm_per_s,
            label=f"U = {wind_speed_m_per_s:.1f} m/s",
        )
    moisture_axes.set_xlabel("fuel moisture content (%)")
    moisture_axes.set_ylabel("rate of spread (mm/s)")
    moisture_axes.set_title("Moisture damping")
    moisture_axes.legend(fontsize="x-small")
    moisture_axes.grid(True, linewidth=0.3, alpha=0.5)

    figure.tight_layout()
    return figure
