"""Ambient air the sound travels through."""

from dataclasses import dataclass
from typing import Any, Self

PASCALS_PER_KILOPASCAL: float = 1000.0


@dataclass(frozen=True)
class AtmosphericConfiguration:
    """Air state feeding the atmospheric absorption model.

    Pressure is carried in pascals here because that is the SI unit a config
    file should speak. `AtmosphericConditions` wants kilopascals, so
    `atmospheric_pressure_kpa` does the conversion at the boundary.

    Attributes:
        air_temperature_celsius: Air temperature, in degrees Celsius.
        relative_humidity_percent: Relative humidity, 0 to 100.
        atmospheric_pressure_pa: Static pressure, in pascals.
    """

    air_temperature_celsius: float = 20.0
    relative_humidity_percent: float = 50.0
    atmospheric_pressure_pa: float = 101325.0

    @property
    def atmospheric_pressure_kpa(self) -> float:
        """Pressure in the unit `AtmosphericConditions` expects.

        Returns:
            Static pressure, in kilopascals.
        """
        return self.atmospheric_pressure_pa / PASCALS_PER_KILOPASCAL

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build from a JSON-decoded mapping, filling absent keys with defaults.

        Args:
            data: Mapping of field name to value.

        Returns:
            The configuration.
        """
        defaults = cls()
        return cls(
            air_temperature_celsius=float(
                data.get("air_temperature_celsius", defaults.air_temperature_celsius)
            ),
            relative_humidity_percent=float(
                data.get(
                    "relative_humidity_percent", defaults.relative_humidity_percent
                )
            ),
            atmospheric_pressure_pa=float(
                data.get("atmospheric_pressure_pa", defaults.atmospheric_pressure_pa)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "air_temperature_celsius": self.air_temperature_celsius,
            "relative_humidity_percent": self.relative_humidity_percent,
            "atmospheric_pressure_pa": self.atmospheric_pressure_pa,
        }
