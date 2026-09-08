"""Protocol satisfied by every fire spread engine in the repository.

Scripts and the simulation context only ever see this type. Swapping the
cellular automaton for the Balbi rate-of-spread engine, or for anything
else, is a config change, not a code change.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fields.vector_field_protocol import VectorFieldProtocol
from src.spark.fire.fire_state import FireState
from src.spark.terrain.mesh_protocol import MeshProtocol


@runtime_checkable
class SpreadEngineProtocol(Protocol):
    """A fire propagation model over a mesh."""

    def initialize(
        self,
        mesh: MeshProtocol,
        fuel_field: ScalarFieldProtocol,
        wind_field: VectorFieldProtocol,
    ) -> FireState:
        """Bind the engine to a mesh and its fields, and return the unburnt state.

        Args:
            mesh: Geometry and connectivity the engine will propagate fire over.
            fuel_field: Fuel available at each position.
            wind_field: Wind vector at each position.

        Returns:
            A FireState with no cell burning, at time 0.
        """
        ...

    def ignite_cells(
        self, state: FireState, cell_indices: npt.NDArray[np.int64]
    ) -> FireState:
        """Start a fire at the given cells, at the state's current time.

        Cells that carry no fuel or have already ignited are skipped. Burnout
        time is set by the engine, which is the only holder of that knowledge.

        Args:
            state: The state to ignite cells in.
            cell_indices: Int64 array of shape (k,), indices of cells to ignite.

        Returns:
            A new FireState at the same time, with the ignitable cells burning.
        """
        ...

    def step(self, state: FireState, dt: float) -> FireState:
        """Advance the simulation by `dt` seconds.

        Args:
            state: The state to advance from.
            dt: Timestep, in seconds.

        Returns:
            A new FireState at time `state.current_time_s + dt`.
        """
        ...
