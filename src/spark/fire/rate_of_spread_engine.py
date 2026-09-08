"""Physically grounded spread engine driven by the Balbi 2009 rate of spread.

Fire crossing one mesh edge is treated as an arrival time problem. When a
cell ignites at time `t`, it schedules an arrival at each neighbor at
`t + distance / R`, where `R` comes from `rate_of_spread_equations.py`
evaluated with the wind and slope resolved along that specific edge. A
cell ignites at the earliest arrival any of its ignited neighbors offers,
which makes ignition times nearly independent of `dt` — the timestep sets
how finely arrivals are resolved, not how far fire travels per step.

A cell keeps its scheduled arrivals after burning out. The front has
already passed through it, and retracting the fire it handed on would
stall spread whenever a cell burns out faster than fire crosses one edge.

Contains no rate of spread physics of its own. Swapping Balbi 2009 for
Balbi 2020 is one call in `_compute_edge_rates_of_spread`.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spark.fields.scalar_field_protocol import ScalarFieldProtocol
from spark.fields.vector_field_protocol import VectorFieldProtocol
from spark.fire.fire_state import FireState
from spark.fire.fuel_properties import FuelProperties
from spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from spark.terrain.mesh_protocol import MeshProtocol

FUEL_LOAD_IGNITION_THRESHOLD_KG_PER_M2 = 1e-6


@dataclass(frozen=True, slots=True)
class _BoundDomain:
    """Mesh, sampled fields and edge crossing times bound by `initialize`."""

    mesh: MeshProtocol
    is_flammable: npt.NDArray[np.bool_]
    edge_crossing_times_s: npt.NDArray[np.float64]
    rate_of_spread_m_per_s: npt.NDArray[np.float64]


class RateOfSpreadEngine:
    """Balbi 2009 driven spread engine, satisfying SpreadEngineProtocol."""

    def __init__(self, fuel: FuelProperties) -> None:
        """Store the fuel bed every cell is assumed to carry.

        Args:
            fuel: Fuel properties used for the whole domain. Per-cell fuel
                heterogeneity replaces this with a lookup against the fuel
                field, without changing the rest of the engine.
        """
        self._fuel = fuel
        self._domain: _BoundDomain | None = None

    def initialize(
        self,
        mesh: MeshProtocol,
        fuel_field: ScalarFieldProtocol,
        wind_field: VectorFieldProtocol,
    ) -> FireState:
        """Bind the engine, precompute every edge crossing time, return unburnt state.

        The crossing times depend only on wind, slope and fuel, none of which
        change during a run, so they are computed once here. A time-varying
        wind field would have to move this into `step`.

        Args:
            mesh: Geometry and connectivity the fire will propagate over.
            fuel_field: Fuel load in kg per square metre at each cell position.
            wind_field: Wind vector in metres per second at each cell position.

        Returns:
            A FireState with no cell burning, at time 0.
        """
        fuel_load_kg_per_m2 = fuel_field.sample(mesh.cell_positions_xyz)
        rate_of_spread_m_per_s = self._compute_edge_rates_of_spread(
            mesh, wind_field.sample(mesh.cell_positions_xyz)
        )
        self._domain = _BoundDomain(
            mesh=mesh,
            is_flammable=fuel_load_kg_per_m2 > FUEL_LOAD_IGNITION_THRESHOLD_KG_PER_M2,
            edge_crossing_times_s=np.where(
                mesh.neighbor_indices >= 0,
                mesh.neighbor_distances_m / rate_of_spread_m_per_s,
                np.inf,
            ),
            rate_of_spread_m_per_s=rate_of_spread_m_per_s,
        )
        return FireState(
            ignition_times_s=np.full(mesh.cell_count, np.inf, dtype=np.float64),
            burnout_times_s=np.full(mesh.cell_count, np.inf, dtype=np.float64),
            is_burning=np.zeros(mesh.cell_count, dtype=np.bool_),
            has_ignited=np.zeros(mesh.cell_count, dtype=np.bool_),
            current_time_s=0.0,
        )

    def _compute_edge_rates_of_spread(
        self, mesh: MeshProtocol, wind_vectors_xyz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Rate of spread along every edge, oriented from neighbor into cell.

        Slot `(n, k)` describes fire arriving at cell `n` from its neighbor
        `k`, so the direction of travel is the reverse of the mesh's stored
        direction. Slope is read off the z component of that unit direction,
        which is how a mesh built over an elevation field carries terrain.

        Args:
            mesh: The mesh supplying directions and connectivity.
            wind_vectors_xyz: Float64 array of shape (cell_count, 3).

        Returns:
            Float64 array of shape (cell_count, max_neighbors), in metres
            per second.
        """
        wind_speed_along_travel_m_per_s = -np.einsum(
            "nkj,nj->nk", mesh.neighbor_unit_directions_xyz, wind_vectors_xyz
        )
        slope_angle_along_travel_rad = np.arcsin(
            np.clip(-mesh.neighbor_unit_directions_xyz[:, :, 2], -1.0, 1.0)
        )
        return compute_rate_of_spread_balbi_2009(
            wind_speed_along_travel_m_per_s, slope_angle_along_travel_rad, self._fuel
        )

    def _require_bound_domain(self) -> _BoundDomain:
        """Return the bound domain, or fail with a message naming the cause.

        Returns:
            The mesh and precomputed edge data bound by `initialize`.

        Raises:
            RuntimeError: If `initialize` has not been called.
        """
        if self._domain is None:
            raise RuntimeError(
                "RateOfSpreadEngine.initialize must be called before igniting "
                "cells or stepping the simulation"
            )
        return self._domain

    @property
    def edge_rates_of_spread_m_per_s(self) -> npt.NDArray[np.float64]:
        """Rate of spread along every mesh edge, for inspection and plotting.

        Returns:
            Float64 array of shape (cell_count, max_neighbors).

        Raises:
            RuntimeError: If `initialize` has not been called.
        """
        return self._require_bound_domain().rate_of_spread_m_per_s

    def ignite_cells(
        self, state: FireState, cell_indices: npt.NDArray[np.int64]
    ) -> FireState:
        """Start a fire at the given cells, at the state's current time.

        Burnout follows the fuel's residence time. Cells with no fuel or
        already ignited are skipped.

        Args:
            state: The state to ignite cells in.
            cell_indices: Int64 array of shape (k,), indices of cells to ignite.

        Returns:
            A new FireState at the same time, with the ignitable cells burning.
        """
        domain = self._require_bound_domain()
        is_requested = np.zeros(domain.mesh.cell_count, dtype=np.bool_)
        is_requested[cell_indices] = True
        newly_ignited = is_requested & domain.is_flammable & ~state.has_ignited
        return self._build_state(
            state,
            newly_ignited,
            np.full_like(state.ignition_times_s, state.current_time_s),
            state.current_time_s,
        )

    def step(self, state: FireState, dt: float) -> FireState:
        """Advance the simulation by `dt` seconds.

        Args:
            state: The state to advance from.
            dt: Timestep, in seconds.

        Returns:
            A new FireState at time `state.current_time_s + dt`.
        """
        domain = self._require_bound_domain()
        new_current_time_s = state.current_time_s + dt
        safe_neighbor_indices = np.where(
            domain.mesh.neighbor_indices >= 0, domain.mesh.neighbor_indices, 0
        )
        earliest_arrival_time_s = (
            state.ignition_times_s[safe_neighbor_indices] + domain.edge_crossing_times_s
        ).min(axis=1)
        newly_ignited = (
            (earliest_arrival_time_s <= new_current_time_s)
            & domain.is_flammable
            & ~state.has_ignited
        )
        return self._build_state(
            state, newly_ignited, earliest_arrival_time_s, new_current_time_s
        )

    def _build_state(
        self,
        state: FireState,
        newly_ignited: npt.NDArray[np.bool_],
        ignition_times_s: npt.NDArray[np.float64],
        new_current_time_s: float,
    ) -> FireState:
        """Fold newly ignited cells into a fresh state and retire burnt ones.

        Args:
            state: The state being advanced.
            newly_ignited: Cells catching fire in this transition.
            ignition_times_s: Candidate ignition time for every cell, read
                only where `newly_ignited` is True.
            new_current_time_s: Time the returned state is stamped with.

        Returns:
            The new FireState.
        """
        next_ignition_times_s = np.where(
            newly_ignited, ignition_times_s, state.ignition_times_s
        )
        next_burnout_times_s = np.where(
            newly_ignited,
            ignition_times_s + self._fuel.residence_time_s,
            state.burnout_times_s,
        )
        next_has_ignited = state.has_ignited | newly_ignited
        return FireState(
            ignition_times_s=next_ignition_times_s,
            burnout_times_s=next_burnout_times_s,
            is_burning=next_has_ignited & (next_burnout_times_s > new_current_time_s),
            has_ignited=next_has_ignited,
            current_time_s=new_current_time_s,
        )
