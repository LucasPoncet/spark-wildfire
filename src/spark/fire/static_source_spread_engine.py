"""A fire that never spreads and never goes out.

Rungs E1 and E2 of the experiment ladder need a source that sits still and
keeps sounding, so that superposition can be tested without motion. Making
that a `SpreadEngineProtocol` implementation rather than a separate render
path means one forward script serves every rung, and the static case gets
the protocol's whole test suite for free.

Burnout is infinite by construction. That is safe for the front mask: an
isolated permanently burning cell keeps un-ignited neighbours forever, so
`compute_fire_front_mask` reports it as on the front at every observation,
and the two emission models agree on it.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fields.vector_field_protocol import VectorFieldProtocol
from src.spark.fire.fire_state import FireState
from src.spark.terrain.mesh_protocol import MeshProtocol

FUEL_LOAD_IGNITION_THRESHOLD_KG_PER_M2: float = 1e-6


@dataclass(frozen=True, slots=True)
class _BoundDomain:
    """Mesh and fuel the engine was initialized against.

    Attributes:
        mesh: The grid the sources live on.
        is_flammable: Bool array of shape (cell_count,).
    """

    mesh: MeshProtocol
    is_flammable: npt.NDArray[np.bool_]


class StaticSourceSpreadEngine:
    """Sources ignite once, burn forever, and never light a neighbour."""

    def __init__(self) -> None:
        """Create an unbound engine. Call `initialize` before anything else."""
        self._domain: _BoundDomain | None = None

    def initialize(
        self,
        mesh: MeshProtocol,
        fuel_field: ScalarFieldProtocol,
        wind_field: VectorFieldProtocol,
    ) -> FireState:
        """Bind the engine to a mesh and return the unlit state.

        The wind field is accepted to satisfy the protocol and ignored: a
        static source does not move, so nothing can advect it.

        Args:
            mesh: Geometry and connectivity, used only for the cell count.
            fuel_field: Fuel available at each position; a cell with none
                cannot be lit.
            wind_field: Ignored.

        Returns:
            A FireState with no cell burning, at time 0.
        """
        del wind_field
        fuel_load_kg_per_m2 = fuel_field.sample(mesh.cell_positions_xyz)
        self._domain = _BoundDomain(
            mesh=mesh,
            is_flammable=fuel_load_kg_per_m2 > FUEL_LOAD_IGNITION_THRESHOLD_KG_PER_M2,
        )
        return FireState(
            ignition_times_s=np.full(mesh.cell_count, np.inf, dtype=np.float64),
            burnout_times_s=np.full(mesh.cell_count, np.inf, dtype=np.float64),
            is_burning=np.zeros(mesh.cell_count, dtype=np.bool_),
            has_ignited=np.zeros(mesh.cell_count, dtype=np.bool_),
            current_time_s=0.0,
        )

    def _require_bound_domain(self) -> _BoundDomain:
        """Return the bound domain, or explain that binding never happened.

        Returns:
            The mesh and fuel mask bound by `initialize`.

        Raises:
            RuntimeError: If `initialize` has not been called.
        """
        if self._domain is None:
            raise RuntimeError(
                "StaticSourceSpreadEngine.initialize must be called before use"
            )
        return self._domain

    def ignite_cells(
        self, state: FireState, cell_indices: npt.NDArray[np.int64]
    ) -> FireState:
        """Light the given cells permanently, at the state's current time.

        Args:
            state: The state to ignite cells in.
            cell_indices: Int64 array of shape (k,), indices of cells to light.

        Returns:
            A new FireState at the same time, with those cells burning forever.
        """
        domain = self._require_bound_domain()
        is_requested = np.zeros(domain.mesh.cell_count, dtype=np.bool_)
        is_requested[cell_indices] = True
        newly_ignited = is_requested & domain.is_flammable & ~state.has_ignited

        ignition_times_s = np.where(
            newly_ignited, state.current_time_s, state.ignition_times_s
        )
        has_ignited = state.has_ignited | newly_ignited
        return FireState(
            ignition_times_s=ignition_times_s,
            burnout_times_s=np.full_like(state.burnout_times_s, np.inf),
            is_burning=has_ignited.copy(),
            has_ignited=has_ignited,
            current_time_s=state.current_time_s,
        )

    def step(self, state: FireState, dt: float) -> FireState:
        """Advance the clock, leaving every mask untouched.

        Args:
            state: The state to advance from.
            dt: Timestep, in seconds.

        Returns:
            A new FireState at time `state.current_time_s + dt`.
        """
        self._require_bound_domain()
        return FireState(
            ignition_times_s=state.ignition_times_s.copy(),
            burnout_times_s=state.burnout_times_s.copy(),
            is_burning=state.is_burning.copy(),
            has_ignited=state.has_ignited.copy(),
            current_time_s=state.current_time_s + dt,
        )
