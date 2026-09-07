from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class MeshProtocol(Protocol):
    """Geometry and connectivity contract shared by every mesh implementation."""

    @property
    def cell_count(self) -> int:
        """Number of cells carried by the mesh.

        Returns:
            Total cell count.
        """
        ...

    @property
    def cell_positions_xyz(self) -> npt.NDArray[np.float64]:
        """Cell centre positions in metres.

        Returns:
            Float64 array of shape (cell_count, 3); z is 0.0 for flat meshes.
        """
        ...

    @property
    def neighbor_indices(self) -> npt.NDArray[np.int64]:
        """Indices of the cells adjacent to each cell.

        Returns:
            Int64 array of shape (cell_count, max_neighbors), -1 in unused slots.
        """
        ...

    @property
    def neighbor_distances_m(self) -> npt.NDArray[np.float64]:
        """Euclidean distance from each cell to each of its neighbors.

        Returns:
            Float64 array of shape (cell_count, max_neighbors), 0.0 where the
            neighbor index is -1.
        """
        ...

    @property
    def neighbor_unit_directions_xyz(self) -> npt.NDArray[np.float64]:
        """Unit vector pointing from each cell towards each of its neighbors.

        Returns:
            Float64 array of shape (cell_count, max_neighbors, 3), zero vector
            where the neighbor index is -1.
        """
        ...

    @property
    def max_neighbors(self) -> int:
        """Width of the neighbor arrays.

        Returns:
            Maximum number of neighbors any cell of the mesh can have.
        """
        ...
