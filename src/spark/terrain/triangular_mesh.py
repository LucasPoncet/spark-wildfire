from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

CARDINAL_TRIANGULAR_NEIGHBOR_OFFSETS_XY: npt.NDArray[np.float64] = np.array(
    [(1,0), (1/2, np.sqrt(3)/2), (-1/2, np.sqrt(3)/2),
      (-1,0), (-1/2, -np.sqrt(3)/2), (1/2, -np.sqrt(3)/2)
    ], dtype=np.float64
)



@dataclass(frozen=True, slots=True)
class TriangularGridMeshConfig:
    """Construction parameters of a regular square grid."""

    extent_x_m: float
    extent_y_m: float
    cell_spacing_m: float


class TriangularGridMesh:
    """Regular square grid with 4- or 8-connectivity, satisfying MeshProtocol."""

    def __init__(self, config: TriangularGridMeshConfig) -> None:
        """Build the grid and precompute every geometry and connectivity array.

        Args:
            config: Extents, spacing and connectivity of the grid.
        """
        self._config = config
        self._n_x = int(config.extent_x_m / config.cell_spacing_m) + 1
        self._n_y = int(2*config.extent_y_m / (np.sqrt(3)*config.cell_spacing_m)) + 1
        self._cell_count = self._n_x *((self._n_y+1)//2) + (self._n_x-1) * ((self._n_y)//2)
        self._neighbor_offsets_xy = (
                    CARDINAL_TRIANGULAR_NEIGHBOR_OFFSETS_XY
                )
        self._cell_positions_xyz = self._build_cell_positions_xyz()
        (
            self._neighbor_indices,
            self._neighbor_distances_m,
            self._neighbor_unit_directions_xyz,
        ) = self._build_neighborhood()

    @classmethod
    def from_values(
        cls,
        extent_x_m: float,
        extent_y_m: float,
        cell_spacing_m: float,
    ) -> Self:
        """Build a mesh without assembling the config first.

        Args:
            extent_x_m: Grid extent along x, in metres.
            extent_y_m: Grid extent along y, in metres.
            cell_spacing_m: Distance between two cardinal neighbors, in metres.
            use_diagonal_neighbors: 8-connectivity if True, 4-connectivity otherwise.

        Returns:
            The constructed mesh.
        """
        return cls(
            TriangularGridMeshConfig(
                extent_x_m=extent_x_m,
                extent_y_m=extent_y_m,
                cell_spacing_m=cell_spacing_m,
            )
        )

    def _build_cell_positions_xyz(self) -> npt.NDArray[np.float64]:
            """Lay the cells out row-major from the origin, x varying fastest.
    
            Returns:
                Float64 array of shape (cell_count, 3) with a zero z column.
            """
            x_m = np.linspace(0.0, self._config.extent_x_m, self._n_x, dtype=np.float64)
            x_m_shift = x_m + self._config.cell_spacing_m/2
            y_m = np.linspace(0.0, self._config.cell_spacing_m*(self._n_y + (-2 + self._n_y%2))* np.sqrt(3)/2 , (self._n_y+1)//2, dtype=np.float64)
            y_m_shift = y_m + np.sqrt(3)*self._config.cell_spacing_m /2 
            grid_x_m, grid_y_m = np.meshgrid(x_m, y_m)
            grid_x_m_shift, grid_y_m_shift = np.meshgrid(x_m_shift, y_m_shift)
            grid_x_m_fusion = np.concatenate([grid_x_m, grid_x_m_shift])
            grid_y_m_fusion = np.concatenate([grid_y_m, grid_y_m_shift])
            grid_x_m_fusion = np.where((0<= grid_x_m_fusion) & (grid_x_m_fusion <= self._config.extent_x_m), grid_x_m_fusion, np.nan)
            grid_y_m_fusion = np.where((0<= grid_y_m_fusion) & (grid_y_m_fusion<= self._config.extent_y_m), grid_y_m_fusion, np.nan)
            stack_grid= np.stack(
                (
                    grid_x_m_fusion.ravel(),
                    grid_y_m_fusion.ravel(),
                    np.zeros(len(grid_x_m_fusion.ravel()), dtype=np.float64),
                ),
                axis=1,
            )
            mask = ~np.isnan(stack_grid).any(axis = 1)
            stack_grid = stack_grid[mask]
            stack_grid = stack_grid[np.lexsort((stack_grid[:,0],stack_grid[:,1]))]
            if self._cell_count != stack_grid.shape[0]:
                raise ValueError(f"Error: cell_count ({self._cell_count}) is not equal to the number of cell in the grid ({stack_grid.shape[0]}.) ")
            return stack_grid
    
    def _build_neighborhood(
            self,
        ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Resolve the neighbors of every cell at once, dropping out-of-grid slots.
    
            Returns:
                Neighbor indices, distances in metres and unit directions, of shapes
                (cell_count, max_neighbors), (cell_count, max_neighbors) and
                (cell_count, max_neighbors, 3).
            """
            neighbor_indices = []
            row = 0
            for i in range(self._cell_count):
                mini_indice_row = row*self._n_x - (row)//2
                mini_indice_previous_row = max((row-1)*self._n_x - (row-1)//2, 0)
                mini_indice_next_row = min((row+1)*self._n_x - (row+1)//2, self._cell_count-1)
                maxi_indice_row = (row + 1)*self._n_x - (row+1)//2 -1 
                maxi_indice_previous_row = max((row)*self._n_x - (row)//2 -1, 0)
                maxi_indice_next_row = min((row+2)*self._n_x - (row+2)//2 -1, self._cell_count-1)

                
                neighbor_indices.append([i-1 if i-1 >= mini_indice_row else -1,
                                       i + self._n_x-1 if( i + self._n_x-1 >= mini_indice_next_row) and (i + self._n_x-1)<self._cell_count else -1,
                                       i + self._n_x if i + self._n_x <= maxi_indice_next_row else -1,
                                       i+1 if i+1 <= maxi_indice_row else -1,
                                       i-self._n_x +1 if (i-self._n_x +1 <= maxi_indice_previous_row) and 0< (i-self._n_x +1) else -1,
                                       i -self._n_x if i -self._n_x >= mini_indice_previous_row else -1
                                       ])
                if i == (row+1)*(self._n_x-1) + (row//2):
                     row +=1
            neighbor_indices = np.array(neighbor_indices)           
            return (
                neighbor_indices,
                np.ones(self._cell_count)*self._config.cell_spacing_m,
                np.zeros(self._cell_count),
            )
    @property
    def config(self) -> TriangularGridMeshConfig:
        """Parameters the mesh was built from.

        Returns:
            The construction config.
        """
        return self._config

    @property
    def n_x(self) -> int:
        """Number of cells along x.

        Returns:
            Cell count of one grid row.
        """
        return self._n_x

    @property
    def n_y(self) -> int:
        """Number of cells along y.

        Returns:
            Cell count of one grid column.
        """
        return self._n_y

    @property
    def cell_count(self) -> int:
        """Number of cells carried by the mesh.

        Returns:
            Total cell count.
        """
        return self._cell_count

    @property
    def cell_positions_xyz(self) -> npt.NDArray[np.float64]:
        """Cell centre positions in metres.

        Returns:
            Float64 array of shape (cell_count, 3); z is 0.0 for this flat mesh.
        """
        return self._cell_positions_xyz

    @property
    def neighbor_indices(self) -> npt.NDArray[np.int64]:
        """Indices of the cells adjacent to each cell.

        Returns:
            Int64 array of shape (cell_count, max_neighbors), -1 in unused slots.
        """
        return self._neighbor_indices

    @property
    def neighbor_distances_m(self) -> npt.NDArray[np.float64]:
        """Euclidean distance from each cell to each of its neighbors.

        Returns:
            Float64 array of shape (cell_count, max_neighbors), 0.0 where the
            neighbor index is -1.
        """
        return self._neighbor_distances_m

    @property
    def neighbor_unit_directions_xyz(self) -> npt.NDArray[np.float64]:
        """Unit vector pointing from each cell towards each of its neighbors.

        Returns:
            Float64 array of shape (cell_count, max_neighbors, 3), zero vector
            where the neighbor index is -1.
        """
        return self._neighbor_unit_directions_xyz

    @property
    def max_neighbors(self) -> int:
        """Width of the neighbor arrays.

        Returns:
            8 with diagonal neighbors enabled, 4 otherwise.
        """
        return 6
    
    def __repr__(self) -> str:
        """Summarise the grid dimensions.

        Returns:
            One-line representation of the mesh.
        """
        return (
            f"TriangularGridMesh(n_x={self._n_x}, n_y={self._n_y}, "
            f"cell_spacing_m={self._config.cell_spacing_m}, "
            f"cell_count={self._cell_count})"
        )