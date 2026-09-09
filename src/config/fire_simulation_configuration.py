"""Fuel choice, ignition point and run length of the fire simulation."""

from dataclasses import dataclass
from typing import Any, Self

FRONT_ONLY_EMISSION: str = "front_only"
ALL_BURNING_EMISSION: str = "all_burning"


@dataclass(frozen=True)
class FireSimulationConfiguration:
    """What burns, where it starts, and for how long.

    Attributes:
        fuel_preset_name: Name of a FuelProperties classmethod preset.
        ignition_cell_x_fraction: Ignition point along x as a fraction of the
            grid extent, 0.0 at the origin corner and 1.0 at the far corner.
        ignition_cell_y_fraction: Same along y.
        simulation_duration_s: Simulated seconds to run for.
        time_step_safety_factor: Fraction of the CFL-limited timestep to use.
        emission_model: Which burning cells radiate, "front_only" for the
            advancing perimeter or "all_burning" for the whole burning area.
    """

    fuel_preset_name: str = "pine_needle_litter"
    ignition_cell_x_fraction: float = 0.5
    ignition_cell_y_fraction: float = 0.5
    simulation_duration_s: float = 120.0
    time_step_safety_factor: float = 0.9
    emission_model: str = FRONT_ONLY_EMISSION

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
            fuel_preset_name=str(
                data.get("fuel_preset_name", defaults.fuel_preset_name)
            ),
            ignition_cell_x_fraction=float(
                data.get("ignition_cell_x_fraction", defaults.ignition_cell_x_fraction)
            ),
            ignition_cell_y_fraction=float(
                data.get("ignition_cell_y_fraction", defaults.ignition_cell_y_fraction)
            ),
            simulation_duration_s=float(
                data.get("simulation_duration_s", defaults.simulation_duration_s)
            ),
            time_step_safety_factor=float(
                data.get("time_step_safety_factor", defaults.time_step_safety_factor)
            ),
            emission_model=str(data.get("emission_model", defaults.emission_model)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "fuel_preset_name": self.fuel_preset_name,
            "ignition_cell_x_fraction": self.ignition_cell_x_fraction,
            "ignition_cell_y_fraction": self.ignition_cell_y_fraction,
            "simulation_duration_s": self.simulation_duration_s,
            "time_step_safety_factor": self.time_step_safety_factor,
            "emission_model": self.emission_model,
        }
