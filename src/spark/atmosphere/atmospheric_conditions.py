import math
from dataclasses import dataclass

KELVIN_AT_ZERO_CELSIUS: float = 273.15
TRIPLE_POINT_TEMPERATURE_K: float = 273.16
REFERENCE_TEMPERATURE_K: float = 293.15
REFERENCE_PRESSURE_KPA: float = 101.325
SPEED_OF_SOUND_COEFFICIENT_M_PER_S_PER_SQRT_K: float = 20.05


@dataclass(frozen=True)
class AtmosphericConditions:
    """Ambient air state for one scenario.

    Attributes:
        air_temperature_celsius: Air temperature in degrees Celsius.
        relative_humidity_percent: Relative humidity in percent.
        pressure_kpa: Static pressure in kilopascals.
    """

    air_temperature_celsius: float
    relative_humidity_percent: float
    pressure_kpa: float = REFERENCE_PRESSURE_KPA


def compute_speed_of_sound_m_per_s(air_temperature_celsius: float) -> float:
    """Computes the speed of sound from air temperature alone.

    Args:
        air_temperature_celsius: Air temperature in degrees Celsius.

    Returns:
        Speed of sound in metres per second, about 340 at 15 degrees Celsius.
    """
    return SPEED_OF_SOUND_COEFFICIENT_M_PER_S_PER_SQRT_K * math.sqrt(
        KELVIN_AT_ZERO_CELSIUS + air_temperature_celsius
    )
