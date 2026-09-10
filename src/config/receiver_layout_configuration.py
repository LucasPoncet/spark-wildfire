"""Where the microphones sit when there are more than two of them."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.array_types import Float64Array

EXPLICIT_LAYOUT: str = "explicit"
RING_LAYOUT: str = "ring"
GRID_LAYOUT: str = "grid"
RANDOM_LAYOUT: str = "random"


@dataclass(frozen=True)
class ReceiverLayoutConfiguration:
    """Receiver layout, as a strategy name plus that strategy's parameters.

    The `explicit` case carries the positions themselves, which is what keeps
    every two-receiver configuration directory working unmodified.

    Attributes:
        layout: One of "explicit", "ring", "grid" or "random".
        count: Number of receivers, used by "ring" and "random".
        explicit_positions_xy_m: Positions of shape `(n_receivers, 2)`, used by
            "explicit".
        ring_radius_m: Ring radius, in metres, used by "ring".
        ring_start_bearing_rad: Bearing of the first receiver, used by "ring".
        center_x_fraction: Layout centre along x as a fraction of the domain
            extent. Shared by "ring" and "random".
        center_y_fraction: Same along y.
        grid_spacing_m: Spacing between receivers, in metres, used by "grid".
        random_radius_m: Sampling disc radius, in metres, used by "random".
        random_seed: Seed making the random layout reproducible.
        minimum_separation_m: Smallest permitted distance between two receivers.
        maximum_collinearity: Largest permitted collinearity measure.
        height_m: Receiver height above the ground plane, in metres.
    """

    layout: str
    count: int
    explicit_positions_xy_m: Float64Array
    ring_radius_m: float
    ring_start_bearing_rad: float
    center_x_fraction: float
    center_y_fraction: float
    grid_spacing_m: float
    random_radius_m: float
    random_seed: int
    minimum_separation_m: float
    maximum_collinearity: float
    height_m: float

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "layout": self.layout,
            "count": self.count,
            "explicit_positions_xy_m": np.asarray(
                self.explicit_positions_xy_m, dtype=np.float64
            ).tolist(),
            "ring_radius_m": self.ring_radius_m,
            "ring_start_bearing_rad": self.ring_start_bearing_rad,
            "center_x_fraction": self.center_x_fraction,
            "center_y_fraction": self.center_y_fraction,
            "grid_spacing_m": self.grid_spacing_m,
            "random_radius_m": self.random_radius_m,
            "random_seed": self.random_seed,
            "minimum_separation_m": self.minimum_separation_m,
            "maximum_collinearity": self.maximum_collinearity,
            "height_m": self.height_m,
        }
