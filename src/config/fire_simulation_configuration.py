"""Fuel choice, ignition points, engine and run length of the fire simulation."""

from dataclasses import dataclass
from typing import Any, Self

FRONT_ONLY_EMISSION: str = "front_only"
ALL_BURNING_EMISSION: str = "all_burning"
DEFAULT_IGNITION_POINTS_XY_FRACTION: tuple[tuple[float, float], ...] = ((0.3, 0.3),)
DEFAULT_TREE_PATCH_CENTERS_XY_FRACTION: tuple[tuple[float, float], ...] = (
    (0.25, 0.25),
    (0.70, 0.60),
    (0.40, 0.80),
)


@dataclass(frozen=True)
class FireSimulationConfiguration:
    """What burns, where it starts, how it spreads, and for how long.

    Attributes:
        fuel_preset_name: Name of a FuelProperties classmethod preset.
        spread_engine_name: Registered name of the propagation model.
        fuel_field_type: Registered name of the scalar field carrying fuel.
        ignition_points_xy_fraction: One or more ignition points, each a
            fraction of the grid extent along x and y, 0.0 at the origin
            corner and 1.0 at the far corner.
        simulation_duration_s: Simulated seconds to run for.
        time_step_safety_factor: Fraction of the CFL-limited timestep to use.
        emission_model: Which burning cells radiate, "front_only" for the
            advancing perimeter or "all_burning" for the whole burning area.
        fuel_moisture_content_fraction: Water carried by the fuel, as a fraction
            of dry mass. Damps the rate of spread.
        tree_density_per_m2: Trees per square metre, used by the tree fuel
            fields. With patches it is the density at a patch centre.
        tree_fuel_load_kg_per_m2: Peak fuel load contributed by one tree.
        tree_influence_radius_m: Gaussian standard deviation of a tree footprint.
        spread_engine_random_seed: Seed making a probabilistic spread engine
            reproducible. The cellular automaton draws a coin per neighbour
            per step and is the only engine that reads it; left unseeded it
            takes its stream from the operating system, so two runs of one
            configuration produce different fires. Measured on `configs/f2`,
            that moved the final-frame sector overlap between 0.28 and 0.68,
            which is wider than most of the effects being measured.
        tree_layout_seed: Seed making a tree layout reproducible.
        tree_patch_centers_xy_fraction: Patch centres as fractions of the grid
            extent, used by the patchy tree field.
        tree_patch_radius_fraction: Patch radius as a fraction of the extent
            along x.
        tree_background_density_per_m2: Tree density far from every patch.
    """

    fuel_preset_name: str = "pine_needle_litter"
    spread_engine_name: str = "rate_of_spread"
    fuel_field_type: str = "uniform"
    ignition_points_xy_fraction: tuple[tuple[float, float], ...] = (
        DEFAULT_IGNITION_POINTS_XY_FRACTION
    )
    simulation_duration_s: float = 1800.0
    time_step_safety_factor: float = 0.9
    emission_model: str = FRONT_ONLY_EMISSION
    fuel_moisture_content_fraction: float = 0.10
    tree_density_per_m2: float = 0.1
    tree_fuel_load_kg_per_m2: float = 0.25
    tree_influence_radius_m: float = 2.5
    spread_engine_random_seed: int = 0
    tree_layout_seed: int = 100
    tree_patch_centers_xy_fraction: tuple[tuple[float, float], ...] = (
        DEFAULT_TREE_PATCH_CENTERS_XY_FRACTION
    )
    tree_patch_radius_fraction: float = 0.15
    tree_background_density_per_m2: float = 0.01

    @property
    def ignition_cell_x_fraction(self) -> float:
        """Position along x of the first ignition point."""
        return self.ignition_points_xy_fraction[0][0]

    @property
    def ignition_cell_y_fraction(self) -> float:
        """Position along y of the first ignition point."""
        return self.ignition_points_xy_fraction[0][1]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build from a decoded mapping, filling absent keys with defaults.

        Accepts either the list form `ignition_points_xy_fraction` or the
        single-point form `ignition_cell_x_fraction` plus
        `ignition_cell_y_fraction`, so every configuration directory and every
        committed `config.json` written before the ladder still loads.

        Args:
            data: Mapping of field name to value.

        Returns:
            The configuration.

        Raises:
            ValueError: If no ignition point is given, or one is not a pair.
        """
        defaults = cls()
        return cls(
            fuel_preset_name=str(
                data.get("fuel_preset_name", defaults.fuel_preset_name)
            ),
            spread_engine_name=str(
                data.get("spread_engine_name", defaults.spread_engine_name)
            ),
            fuel_field_type=str(data.get("fuel_field_type", defaults.fuel_field_type)),
            ignition_points_xy_fraction=read_ignition_points_xy_fraction(
                data, defaults.ignition_points_xy_fraction
            ),
            simulation_duration_s=float(
                data.get("simulation_duration_s", defaults.simulation_duration_s)
            ),
            time_step_safety_factor=float(
                data.get("time_step_safety_factor", defaults.time_step_safety_factor)
            ),
            emission_model=str(data.get("emission_model", defaults.emission_model)),
            fuel_moisture_content_fraction=float(
                data.get(
                    "fuel_moisture_content_fraction",
                    defaults.fuel_moisture_content_fraction,
                )
            ),
            tree_density_per_m2=float(
                data.get("tree_density_per_m2", defaults.tree_density_per_m2)
            ),
            tree_fuel_load_kg_per_m2=float(
                data.get("tree_fuel_load_kg_per_m2", defaults.tree_fuel_load_kg_per_m2)
            ),
            tree_influence_radius_m=float(
                data.get("tree_influence_radius_m", defaults.tree_influence_radius_m)
            ),
            spread_engine_random_seed=int(
                data.get(
                    "spread_engine_random_seed", defaults.spread_engine_random_seed
                )
            ),
            tree_layout_seed=int(
                data.get("tree_layout_seed", defaults.tree_layout_seed)
            ),
            tree_patch_centers_xy_fraction=read_fraction_pairs(
                data.get("tree_patch_centers_xy_fraction"),
                defaults.tree_patch_centers_xy_fraction,
                "tree patch centre",
            ),
            tree_patch_radius_fraction=float(
                data.get(
                    "tree_patch_radius_fraction", defaults.tree_patch_radius_fraction
                )
            ),
            tree_background_density_per_m2=float(
                data.get(
                    "tree_background_density_per_m2",
                    defaults.tree_background_density_per_m2,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "fuel_preset_name": self.fuel_preset_name,
            "spread_engine_name": self.spread_engine_name,
            "fuel_field_type": self.fuel_field_type,
            "ignition_points_xy_fraction": [
                list(point) for point in self.ignition_points_xy_fraction
            ],
            "simulation_duration_s": self.simulation_duration_s,
            "time_step_safety_factor": self.time_step_safety_factor,
            "emission_model": self.emission_model,
            "fuel_moisture_content_fraction": self.fuel_moisture_content_fraction,
            "tree_density_per_m2": self.tree_density_per_m2,
            "tree_fuel_load_kg_per_m2": self.tree_fuel_load_kg_per_m2,
            "tree_influence_radius_m": self.tree_influence_radius_m,
            "spread_engine_random_seed": self.spread_engine_random_seed,
            "tree_layout_seed": self.tree_layout_seed,
            "tree_patch_centers_xy_fraction": [
                list(point) for point in self.tree_patch_centers_xy_fraction
            ],
            "tree_patch_radius_fraction": self.tree_patch_radius_fraction,
            "tree_background_density_per_m2": self.tree_background_density_per_m2,
        }


def read_ignition_points_xy_fraction(
    data: dict[str, Any],
    default_points_xy_fraction: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Read ignition points from either the list form or the scalar pair form.

    Args:
        data: Mapping of field name to value.
        default_points_xy_fraction: Points to use when neither form is present.

    Returns:
        One or more `(x_fraction, y_fraction)` pairs.

    Raises:
        ValueError: If the list is empty, or an entry is not a pair.
    """
    if "ignition_points_xy_fraction" in data:
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in data["ignition_points_xy_fraction"]
            if len(point) == 2
        )
        if len(points) != len(data["ignition_points_xy_fraction"]):
            raise ValueError("every ignition point must be an [x, y] fraction pair")
        if not points:
            raise ValueError("at least one ignition point is required")
        return points

    if "ignition_cell_x_fraction" in data or "ignition_cell_y_fraction" in data:
        return (
            (
                float(
                    data.get(
                        "ignition_cell_x_fraction", default_points_xy_fraction[0][0]
                    )
                ),
                float(
                    data.get(
                        "ignition_cell_y_fraction", default_points_xy_fraction[0][1]
                    )
                ),
            ),
        )

    return default_points_xy_fraction


type RawPairs = list[list[float]] | list[tuple[float, float]] | None


def read_fraction_pairs(
    raw_pairs: RawPairs,
    default_pairs: tuple[tuple[float, float], ...],
    what: str,
) -> tuple[tuple[float, float], ...]:
    """Read a list of `[x, y]` fraction pairs, or fall back to a default.

    Args:
        raw_pairs: The decoded value, or None when the key is absent.
        default_pairs: Pairs to use when the key is absent.
        what: Name of the quantity, used in the error message.

    Returns:
        One or more `(x_fraction, y_fraction)` pairs.

    Raises:
        ValueError: If the list is empty, or an entry is not a pair.
    """
    if raw_pairs is None:
        return default_pairs
    pairs = tuple(
        (float(pair[0]), float(pair[1])) for pair in raw_pairs if len(pair) == 2
    )
    if len(pairs) != len(raw_pairs):
        raise ValueError(f"every {what} must be an [x, y] fraction pair")
    if not pairs:
        raise ValueError(f"at least one {what} is required")
    return pairs
