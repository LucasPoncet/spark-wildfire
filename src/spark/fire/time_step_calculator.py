"""Largest timestep that still resolves the fire front cell by cell.

The front advances at R metres per second. If a timestep is longer than the
time fire needs to cross the shortest mesh edge, the front skips cells and
the result starts depending on the timestep instead of on the physics. This
is the discrete CFL condition applied to front tracking.
"""

import numpy as np

from src.spark.terrain.mesh_protocol import MeshProtocol


def compute_maximum_stable_time_step_s(
    mesh: MeshProtocol,
    maximum_rate_of_spread_m_per_s: float,
    safety_factor: float = 0.9,
    residence_time_s: float | None = None,
) -> float:
    """Time for fire to cross the shortest mesh edge, scaled by a safety factor.

    Args:
        mesh: Mesh supplying neighbor distances and connectivity.
        maximum_rate_of_spread_m_per_s: Fastest rate of spread expected
            anywhere in the domain.
        safety_factor: Fraction of the limiting timestep to return, below 1.0.
        residence_time_s: How long a cell stays alight. When given, the
            timestep is also capped so a cell cannot ignite and burn out
            between two samples, which would hide it from the renderer.

    Returns:
        The maximum stable timestep, in seconds.

    Raises:
        ValueError: If the rate of spread is not positive, or if the mesh has
            no valid neighbors to measure.
    """
    if maximum_rate_of_spread_m_per_s <= 0.0:
        raise ValueError("maximum rate of spread must be positive")

    is_valid_neighbor = mesh.neighbor_indices >= 0
    if not np.any(is_valid_neighbor):
        raise ValueError("mesh has no connected cells to derive a timestep from")

    minimum_neighbor_distance_m = float(
        np.min(mesh.neighbor_distances_m[is_valid_neighbor])
    )
    crossing_limited_time_step_s = (
        safety_factor * minimum_neighbor_distance_m / maximum_rate_of_spread_m_per_s
    )
    if residence_time_s is None:
        return crossing_limited_time_step_s
    return min(crossing_limited_time_step_s, safety_factor * residence_time_s)
