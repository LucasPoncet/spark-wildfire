"""Uniform wind driving the fire front."""

from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class WindConfiguration:
    """Speed and direction of the constant wind field.

    Attributes:
        wind_speed_m_per_s: Wind speed, in metres per second.
        wind_bearing_rad: Direction the wind blows towards, in radians,
            counter-clockwise from the positive x-axis.
    """

    wind_speed_m_per_s: float = 2.0
    wind_bearing_rad: float = 0.0

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
            wind_speed_m_per_s=float(
                data.get("wind_speed_m_per_s", defaults.wind_speed_m_per_s)
            ),
            wind_bearing_rad=float(
                data.get("wind_bearing_rad", defaults.wind_bearing_rad)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "wind_speed_m_per_s": self.wind_speed_m_per_s,
            "wind_bearing_rad": self.wind_bearing_rad,
        }
