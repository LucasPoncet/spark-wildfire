"""Measured properties of one fuel bed, plus named presets from published tables.

Holds raw measured inputs only. Every derived quantity — packing ratio,
optical depth, base rate of spread — is a pure function in
`rate_of_spread_equations.py`, so this file never computes anything.
"""

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class FuelProperties:
    """One fuel bed, as measured in the laboratory.

    Attributes:
        fuel_density_kg_per_m3: Density of the solid fuel particles.
        moisture_content_fraction: Water mass over dry fuel mass, 0.10 for
            10 per cent. Equations that need it as a percentage convert it
            themselves.
        surface_area_to_volume_ratio_per_m: Particle surface area per unit
            volume, the fineness of the fuel.
        fuel_load_kg_per_m2: Dry fuel mass per unit ground area.
        residence_time_s: How long a point on the ground stays aflame.
        fuel_bed_depth_m: Thickness of the fuel layer above the ground.
    """

    fuel_density_kg_per_m3: float
    moisture_content_fraction: float
    surface_area_to_volume_ratio_per_m: float
    fuel_load_kg_per_m2: float
    residence_time_s: float
    fuel_bed_depth_m: float

    @classmethod
    def pine_needle_litter(cls, moisture_content_fraction: float = 0.10) -> Self:
        """Pinus pinaster needle litter, Balbi 2009 Table 1 and Table 2 case 1.

        Args:
            moisture_content_fraction: Water content to build the preset at.
                Table 2 tabulates 0.10 and 0.18.

        Returns:
            The preset fuel bed at the requested moisture.
        """
        return cls(
            fuel_density_kg_per_m3=680.0,
            moisture_content_fraction=moisture_content_fraction,
            surface_area_to_volume_ratio_per_m=4550.0,
            fuel_load_kg_per_m2=0.5,
            residence_time_s=20.0,
            fuel_bed_depth_m=0.04,
        )
