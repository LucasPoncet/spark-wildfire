import numpy as np

from src.spark.atmosphere.atmospheric_conditions import (
    KELVIN_AT_ZERO_CELSIUS,
    REFERENCE_PRESSURE_KPA,
    REFERENCE_TEMPERATURE_K,
    TRIPLE_POINT_TEMPERATURE_K,
    AtmosphericConditions,
)
from src.utils.array_types import Float64Array

NEPER_TO_DECIBEL: float = 8.686
OXYGEN_RELAXATION_STRENGTH: float = 0.01275
NITROGEN_RELAXATION_STRENGTH: float = 0.1068
OXYGEN_RELAXATION_TEMPERATURE_K: float = 2239.1
NITROGEN_RELAXATION_TEMPERATURE_K: float = 3352.0
CLASSICAL_ABSORPTION_COEFFICIENT: float = 1.84e-11


def compute_absorption_coefficients_db_per_m(
    frequencies_hz: Float64Array | float,
    conditions: AtmosphericConditions,
) -> Float64Array:
    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    temperature_k = conditions.air_temperature_celsius + KELVIN_AT_ZERO_CELSIUS
    temperature_ratio = temperature_k / REFERENCE_TEMPERATURE_K
    pressure_ratio = REFERENCE_PRESSURE_KPA / conditions.pressure_kpa

    saturation_pressure_ratio = 10.0 ** (
        -6.8346 * (TRIPLE_POINT_TEMPERATURE_K / temperature_k) ** 1.261 + 4.6151
    )
    molar_water_vapour_percent = (
        conditions.relative_humidity_percent
        * saturation_pressure_ratio
        * pressure_ratio
    )

    oxygen_relaxation_frequency_hz = (1.0 / pressure_ratio) * (
        24.0
        + 4.04e4
        * molar_water_vapour_percent
        * (0.02 + molar_water_vapour_percent)
        / (0.391 + molar_water_vapour_percent)
    )
    nitrogen_relaxation_frequency_hz = (
        (1.0 / pressure_ratio)
        * temperature_ratio**-0.5
        * (
            9.0
            + 280.0
            * molar_water_vapour_percent
            * np.exp(-4.170 * (temperature_ratio ** (-1.0 / 3.0) - 1.0))
        )
    )

    classical_term = (
        CLASSICAL_ABSORPTION_COEFFICIENT * pressure_ratio * temperature_ratio**0.5
    )
    oxygen_term = (
        OXYGEN_RELAXATION_STRENGTH
        * np.exp(-OXYGEN_RELAXATION_TEMPERATURE_K / temperature_k)
        / (
            oxygen_relaxation_frequency_hz
            + frequency**2 / oxygen_relaxation_frequency_hz
        )
    )
    nitrogen_term = (
        NITROGEN_RELAXATION_STRENGTH
        * np.exp(-NITROGEN_RELAXATION_TEMPERATURE_K / temperature_k)
        / (
            nitrogen_relaxation_frequency_hz
            + frequency**2 / nitrogen_relaxation_frequency_hz
        )
    )

    return (
        NEPER_TO_DECIBEL
        * frequency**2
        * (classical_term + temperature_ratio**-2.5 * (oxygen_term + nitrogen_term))
    )
