"""Balbi 2009 rate of spread model, as pure vectorized functions.

The only file where paper notation is allowed, and only inside the body of
a function whose name states which equation it implements.

Two families live here. Startup functions take a `FuelProperties` and
return one float per fuel bed; they are evaluated once, not per cell. Per
cell functions take arrays of flame tilt and return arrays, and are
evaluated once per timestep for every cell at once.

Source: Balbi et al. (2009), A physical model for wildland fires,
Combustion and Flame 156(12). doi:10.1016/j.combustflame.2009.07.010
"""

import numpy as np
import numpy.typing as npt

from spark.fire.fuel_properties import FuelProperties

AMBIENT_TEMPERATURE_K = 300.0
IGNITION_TEMPERATURE_K = 600.0
LATENT_HEAT_OF_EVAPORATION_J_PER_KG = 2.3e6
FUEL_SPECIFIC_HEAT_J_PER_KG_K = 2000.0
MOISTURE_WEIGHTING_COEFFICIENT = 0.05
BASE_MASS_FLUX_KG_PER_M2_S = 0.05
BASE_VELOCITY_COEFFICIENT_M3_PER_KG = 80.0
RADIANT_FRACTION_ROS_MULTIPLIER = 12.0
DRY_RADIATIVE_COEFFICIENT = 2.25
MAXIMUM_FLAME_TILT_ANGLE_RAD = 1.4


def compute_flame_tilt_angle_rad(
    wind_speed_along_front_normal_m_per_s: npt.NDArray[np.float64],
    slope_angle_along_front_normal_rad: npt.NDArray[np.float64],
    upward_gas_velocity_m_per_s: float,
) -> npt.NDArray[np.float64]:
    """Angle the flame leans away from vertical. Balbi 2009 Eq. 2.

    Both inputs are already projected onto the front normal, which reduces
    `tan γ = tan α cos φ + (U / u₀) cos ψ` to a sum of two scalars per cell.
    The result is clamped to `MAXIMUM_FLAME_TILT_ANGLE_RAD`, beyond which
    assumption H2 of the paper fails and the closed form diverges.

    Args:
        wind_speed_along_front_normal_m_per_s: Wind component along the
            direction of spread. Negative against the spread direction.
        slope_angle_along_front_normal_rad: Terrain slope angle along the
            direction of spread. Negative downhill.
        upward_gas_velocity_m_per_s: The fuel bed's u₀, from
            `compute_upward_gas_velocity_m_per_s`.

    Returns:
        Flame tilt angle in radians, same shape as the inputs. Zero or
        negative marks a backing fire.
    """
    tan_gamma = (
        np.tan(slope_angle_along_front_normal_rad)
        + wind_speed_along_front_normal_m_per_s / upward_gas_velocity_m_per_s
    )
    return np.clip(
        np.arctan(tan_gamma),
        -MAXIMUM_FLAME_TILT_ANGLE_RAD,
        MAXIMUM_FLAME_TILT_ANGLE_RAD,
    )


def compute_packing_ratio(fuel: FuelProperties) -> float:
    """Fraction of the fuel bed volume occupied by solid fuel. Balbi 2009 Table 2.

    Args:
        fuel: The fuel bed.

    Returns:
        β, dimensionless.
    """
    return fuel.fuel_load_kg_per_m2 / (
        fuel.fuel_density_kg_per_m3 * fuel.fuel_bed_depth_m
    )


def compute_optical_depth_m(fuel: FuelProperties) -> float:
    """Depth over which the fuel bed absorbs radiation. Balbi 2009 Table 2.

    Args:
        fuel: The fuel bed.

    Returns:
        δ, in metres.
    """
    return 4.0 / (fuel.surface_area_to_volume_ratio_per_m * compute_packing_ratio(fuel))


def compute_absorption_coefficient(fuel: FuelProperties) -> float:
    """Share of incident radiation the bed absorbs. Balbi 2009 Table 2.

    A bed deeper than its own optical depth absorbs everything, hence the cap.

    Args:
        fuel: The fuel bed.

    Returns:
        μ, dimensionless, at most 1.0.
    """
    return min(fuel.fuel_bed_depth_m / compute_optical_depth_m(fuel), 1.0)


def compute_upward_gas_velocity_m_per_s(fuel: FuelProperties) -> float:
    """Speed of the gas rising from the fuel bed on flat ground. Balbi 2009 Eq. 15.

    Args:
        fuel: The fuel bed.

    Returns:
        u₀, in metres per second.
    """
    return (
        BASE_VELOCITY_COEFFICIENT_M3_PER_KG
        * fuel.fuel_load_kg_per_m2
        / fuel.residence_time_s
    )


def compute_ignition_energy_j_per_kg(fuel: FuelProperties) -> float:
    """Energy needed to bring one kilogram of fuel to ignition.

    Heats the dry fuel from ambient to ignition temperature, then evaporates
    the water it carries. This is the physical form; the moisture weighting
    used by `compute_base_rate_of_spread_m_per_s` and
    `compute_radiative_coefficient` instead uses the value Balbi 2009 §2.5
    fits to experiment, which is not this ratio.

    Args:
        fuel: The fuel bed.

    Returns:
        q, in joules per kilogram.
    """
    return (
        FUEL_SPECIFIC_HEAT_J_PER_KG_K * (IGNITION_TEMPERATURE_K - AMBIENT_TEMPERATURE_K)
        + fuel.moisture_content_fraction * LATENT_HEAT_OF_EVAPORATION_J_PER_KG
    )


def compute_moisture_damping_factor(fuel: FuelProperties) -> float:
    """Factor by which water in the fuel slows the fire. Balbi 2009 Eq. 14 and 15.

    The paper writes this as `1 + a m` with `a = 0.05` fitted in §2.5 and `m`
    expressed as a percentage, not as a fraction.

    Args:
        fuel: The fuel bed.

    Returns:
        The dimensionless divisor, 1.0 for oven-dry fuel.
    """
    moisture_content_percent = fuel.moisture_content_fraction * 100.0
    return 1.0 + MOISTURE_WEIGHTING_COEFFICIENT * moisture_content_percent


def compute_base_rate_of_spread_m_per_s(fuel: FuelProperties) -> float:
    """Rate of spread with no wind and no slope. Balbi 2009 Eq. 15.

    Args:
        fuel: The fuel bed.

    Returns:
        R₀, in metres per second.
    """
    return (
        (fuel.fuel_bed_depth_m / fuel.fuel_load_kg_per_m2)
        * BASE_MASS_FLUX_KG_PER_M2_S
        / compute_moisture_damping_factor(fuel)
    )


def compute_radiative_coefficient(fuel: FuelProperties) -> float:
    """Strength of radiative preheating ahead of the front. Balbi 2009 Eq. 14.

    Args:
        fuel: The fuel bed.

    Returns:
        A, dimensionless.
    """
    return (
        compute_absorption_coefficient(fuel)
        * DRY_RADIATIVE_COEFFICIENT
        / compute_moisture_damping_factor(fuel)
    )


def compute_radiant_fraction_velocity_m_per_s(fuel: FuelProperties) -> float:
    """Velocity scale of the radiant fraction of the flame. Balbi 2009 §2.3 E7.

    Args:
        fuel: The fuel bed.

    Returns:
        ν₀, in metres per second.
    """
    return RADIANT_FRACTION_ROS_MULTIPLIER * compute_base_rate_of_spread_m_per_s(fuel)


def _solve_rate_of_spread_m_per_s(
    flame_tilt_angle_rad: npt.NDArray[np.float64], fuel: FuelProperties
) -> npt.NDArray[np.float64]:
    """Positive root of the Balbi 2009 rate of spread quadratic. Eq. 11a and 11b.

    Eq. 13b states the model implicitly in `r = R / R₀`. Rearranged it is a
    quadratic in r whose positive root is this closed form, so no iteration
    is needed. A flame tilted backwards or not at all carries no radiation
    forward and the front creeps at R₀ (Eq. 13a).

    Args:
        flame_tilt_angle_rad: γ, one entry per cell or cell-neighbor pair.
        fuel: The fuel bed being burnt.

    Returns:
        Rate of spread in metres per second, same shape as the input.
    """
    base_rate_of_spread_m_per_s = compute_base_rate_of_spread_m_per_s(fuel)
    radiant_fraction_velocity_m_per_s = compute_radiant_fraction_velocity_m_per_s(fuel)
    radiative_coefficient = compute_radiative_coefficient(fuel)
    cos_gamma = np.cos(flame_tilt_angle_rad)
    sin_gamma = np.sin(flame_tilt_angle_rad)

    linear_term = (
        base_rate_of_spread_m_per_s
        + radiative_coefficient
        * radiant_fraction_velocity_m_per_s
        / cos_gamma
        * (1.0 + sin_gamma - cos_gamma)
        - radiant_fraction_velocity_m_per_s / cos_gamma
    )
    heading_rate_of_spread_m_per_s = 0.5 * (
        linear_term
        + np.sqrt(
            linear_term**2
            + 4.0
            * radiant_fraction_velocity_m_per_s
            * base_rate_of_spread_m_per_s
            / cos_gamma
        )
    )
    return np.where(
        flame_tilt_angle_rad > 0.0,
        heading_rate_of_spread_m_per_s,
        base_rate_of_spread_m_per_s,
    )


def compute_rate_of_spread_balbi_2009(
    wind_speed_along_front_normal_m_per_s: npt.NDArray[np.float64],
    slope_angle_along_front_normal_rad: npt.NDArray[np.float64],
    fuel: FuelProperties,
) -> npt.NDArray[np.float64]:
    """Rate of spread of the fire front. Balbi 2009 Eq. 2 then Eq. 11a and 11b.

    Args:
        wind_speed_along_front_normal_m_per_s: Wind component along the
            direction of spread, one entry per cell or cell-neighbor pair.
        slope_angle_along_front_normal_rad: Terrain slope angle along the
            direction of spread, same shape.
        fuel: The fuel bed being burnt.

    Returns:
        Rate of spread in metres per second, same shape as the inputs.
    """
    return _solve_rate_of_spread_m_per_s(
        compute_flame_tilt_angle_rad(
            wind_speed_along_front_normal_m_per_s,
            slope_angle_along_front_normal_rad,
            compute_upward_gas_velocity_m_per_s(fuel),
        ),
        fuel,
    )


def compute_reduced_rate_of_spread_balbi_2009(
    flame_tilt_angle_rad: npt.NDArray[np.float64], fuel: FuelProperties
) -> npt.NDArray[np.float64]:
    """Rate of spread as a multiple of R₀, given the tilt. Balbi 2009 Eq. 13.

    The universal curve of Fig. 6, reachable without going through a wind
    speed and the fuel's u₀.

    Args:
        flame_tilt_angle_rad: γ, one entry per sample.
        fuel: The fuel bed being burnt.

    Returns:
        r = R / R₀, dimensionless, same shape as the input.
    """
    return _solve_rate_of_spread_m_per_s(
        flame_tilt_angle_rad, fuel
    ) / compute_base_rate_of_spread_m_per_s(fuel)
