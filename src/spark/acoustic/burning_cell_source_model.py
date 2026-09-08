"""Turns a FireState into a set of acoustic point sources.

The only bridge between the fire domain and the acoustic domain. Every
burning cell becomes one source at its own cell position. Knows nothing
about receivers, channels or propagation — a channel takes the positions
this returns and answers with a gain matrix.

Day one gives every burning cell the same amplitude, so all fires sound
alike and only geometry drives the received levels. The physics-based
upgrade replaces the constant with a per-cell mass loss rate once the
Balbi engine writes one into FireState.
"""

import numpy as np
import numpy.typing as npt

from spark.fire.fire_state import FireState
from spark.terrain.mesh_protocol import MeshProtocol


class BurningCellSourceModel:
    """Every burning cell is one acoustic source of constant amplitude."""

    def __init__(self, mesh: MeshProtocol, source_amplitude: float = 1.0) -> None:
        """Bind the model to the mesh whose cells will become sources.

        Args:
            mesh: Geometry supplying the position of each cell.
            source_amplitude: Amplitude assigned to every burning cell.
        """
        self._mesh = mesh
        self._source_amplitude = source_amplitude

    def compute_sources(
        self, state: FireState
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Collect the positions and amplitudes of every currently burning cell.

        Args:
            state: The fire state to read burning cells from.

        Returns:
            A tuple of positions, float64 of shape (source_count, 3), and
            amplitudes, float64 of shape (source_count,). Both are empty
            when nothing is burning.
        """
        burning_cell_indices = np.flatnonzero(state.is_burning)
        return (
            self._mesh.cell_positions_xyz[burning_cell_indices],
            np.full(
                burning_cell_indices.size, self._source_amplitude, dtype=np.float64
            ),
        )
