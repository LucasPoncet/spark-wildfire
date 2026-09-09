"""Where the microphones sit."""

from dataclasses import dataclass
from typing import Any, Self

RING_PLACEMENT: str = "ring"
GRID_PLACEMENT: str = "grid"
RANDOM_PLACEMENT: str = "random"


@dataclass(frozen=True)
class ReceiverConfiguration:
    """Receiver layout, as a strategy name plus that strategy's parameters.

    Attributes:
        placement_strategy: One of "ring", "grid" or "random".
        receiver_count: Number of receivers, used by "ring" and "random".
        ring_radius_m: Ring radius, in metres, used by "ring".
        ring_center_x_fraction: Layout centre along x as a fraction of the
            grid extent. Shared by "ring" and "random".
        ring_center_y_fraction: Same along y.
        grid_spacing_m: Spacing between receivers, in metres, used by "grid".
        random_radius_m: Sampling disc radius, in metres, used by "random".
        random_seed: Seed making the random layout reproducible.
    """

    placement_strategy: str = RING_PLACEMENT
    receiver_count: int = 8
    ring_radius_m: float = 40.0
    ring_center_x_fraction: float = 0.5
    ring_center_y_fraction: float = 0.5
    grid_spacing_m: float = 10.0
    random_radius_m: float = 40.0
    random_seed: int = 0

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
            placement_strategy=str(
                data.get("placement_strategy", defaults.placement_strategy)
            ),
            receiver_count=int(data.get("receiver_count", defaults.receiver_count)),
            ring_radius_m=float(data.get("ring_radius_m", defaults.ring_radius_m)),
            ring_center_x_fraction=float(
                data.get("ring_center_x_fraction", defaults.ring_center_x_fraction)
            ),
            ring_center_y_fraction=float(
                data.get("ring_center_y_fraction", defaults.ring_center_y_fraction)
            ),
            grid_spacing_m=float(data.get("grid_spacing_m", defaults.grid_spacing_m)),
            random_radius_m=float(
                data.get("random_radius_m", defaults.random_radius_m)
            ),
            random_seed=int(data.get("random_seed", defaults.random_seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "placement_strategy": self.placement_strategy,
            "receiver_count": self.receiver_count,
            "ring_radius_m": self.ring_radius_m,
            "ring_center_x_fraction": self.ring_center_x_fraction,
            "ring_center_y_fraction": self.ring_center_y_fraction,
            "grid_spacing_m": self.grid_spacing_m,
            "random_radius_m": self.random_radius_m,
            "random_seed": self.random_seed,
        }
