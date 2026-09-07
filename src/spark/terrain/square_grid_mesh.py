"""Regular square grid mesh with 4- or 8-connectivity.

Example:
    from spark.terrain.square_grid_mesh import SquareGridMesh, SquareGridMeshConfig

    mesh = SquareGridMesh(
        SquareGridMeshConfig(
            extent_x_m=100.0,
            extent_y_m=100.0,
            cell_spacing_m=0.5,
            use_diagonal_neighbors=False,
        )
    )
    mesh = SquareGridMesh.from_values(100.0, 100.0, 0.5, False)

    is_valid = mesh.neighbor_indices >= 0
    mean_spacing_m = mesh.neighbor_distances_m[is_valid].mean()

Cells are laid out row-major from the origin with x varying fastest: cell
row * n_x + column sits at (column * spacing, row * spacing, 0.0), so a per-cell
array reshapes to a 2D image with values.reshape(n_y, n_x).

Neighbor rows are padded to max_neighbors with -1, with matching zeros in
neighbor_distances_m and neighbor_unit_directions_xyz, so mask before use.

Consumers annotate against MeshProtocol rather than against this class.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

CARDINAL_NEIGHBOR_OFFSETS_XY: npt.NDArray[np.int64] = np.array(
    [(1, 0), (0, 1), (-1, 0), (0, -1)], dtype=np.int64
)
DIAGONAL_NEIGHBOR_OFFSETS_XY: npt.NDArray[np.int64] = np.array(
    [(1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=np.int64
)


@dataclass(frozen=True, slots=True)
class SquareGridMeshConfig:
    """Construction parameters of a regular square grid."""

    extent_x_m: float
    extent_y_m: float
    cell_spacing_m: float
    use_diagonal_neighbors: bool


class SquareGridMesh:
    """Regular square grid with 4- or 8-connectivity, satisfying MeshProtocol."""

    def __init__(self, config: SquareGridMeshConfig) -> None:
        """Build the grid and precompute every geometry and connectivity array.

        Args:
            config: Extents, spacing and connectivity of the grid.
        """
        self._config = config
        self._n_x = round(config.extent_x_m / config.cell_spacing_m) + 1
        self._n_y = round(config.extent_y_m / config.cell_spacing_m) + 1
        self._cell_count = self._n_x * self._n_y
        self._neighbor_offsets_xy = (
            np.concatenate((CARDINAL_NEIGHBOR_OFFSETS_XY, DIAGONAL_NEIGHBOR_OFFSETS_XY))
            if config.use_diagonal_neighbors
            else CARDINAL_NEIGHBOR_OFFSETS_XY
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
        use_diagonal_neighbors: bool,
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
            SquareGridMeshConfig(
                extent_x_m=extent_x_m,
                extent_y_m=extent_y_m,
                cell_spacing_m=cell_spacing_m,
                use_diagonal_neighbors=use_diagonal_neighbors,
            )
        )

    def _build_cell_positions_xyz(self) -> npt.NDArray[np.float64]:
        """Lay the cells out row-major from the origin, x varying fastest.

        Returns:
            Float64 array of shape (cell_count, 3) with a zero z column.
        """
        x_m = np.linspace(0.0, self._config.extent_x_m, self._n_x, dtype=np.float64)
        y_m = np.linspace(0.0, self._config.extent_y_m, self._n_y, dtype=np.float64)
        grid_x_m, grid_y_m = np.meshgrid(x_m, y_m)
        return np.stack(
            (
                grid_x_m.ravel(),
                grid_y_m.ravel(),
                np.zeros(self._cell_count, dtype=np.float64),
            ),
            axis=1,
        )

    def _build_neighborhood(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Resolve the neighbors of every cell at once, dropping out-of-grid slots.

        Returns:
            Neighbor indices, distances in metres and unit directions, of shapes
            (cell_count, max_neighbors), (cell_count, max_neighbors) and
            (cell_count, max_neighbors, 3).
        """
        cell_indices = np.arange(self._cell_count, dtype=np.int64)
        column_indices = cell_indices % self._n_x
        row_indices = cell_indices // self._n_x
        offsets_xy = self._neighbor_offsets_xy
        neighbor_columns = column_indices[:, None] + offsets_xy[None, :, 0]
        neighbor_rows = row_indices[:, None] + offsets_xy[None, :, 1]
        is_inside_grid = (
            (neighbor_columns >= 0)
            & (neighbor_columns < self._n_x)
            & (neighbor_rows >= 0)
            & (neighbor_rows < self._n_y)
        )
        flat_indices = neighbor_rows * self._n_x + neighbor_columns
        neighbor_indices = np.where(is_inside_grid, flat_indices, -1)
        displacements_xyz = (
            self._cell_positions_xyz[np.where(is_inside_grid, flat_indices, 0)]
            - self._cell_positions_xyz[:, None, :]
        )
        distances_m = np.linalg.norm(displacements_xyz, axis=2)
        unit_directions_xyz = np.zeros_like(displacements_xyz)
        np.divide(
            displacements_xyz,
            distances_m[:, :, None],
            out=unit_directions_xyz,
            where=is_inside_grid[:, :, None],
        )
        return (
            neighbor_indices,
            np.where(is_inside_grid, distances_m, 0.0),
            unit_directions_xyz,
        )

    @property
    def config(self) -> SquareGridMeshConfig:
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
        return len(self._neighbor_offsets_xy)

    def __repr__(self) -> str:
        """Summarise the grid dimensions.

        Returns:
            One-line representation of the mesh.
        """
        return (
            f"SquareGridMesh(n_x={self._n_x}, n_y={self._n_y}, "
            f"cell_spacing_m={self._config.cell_spacing_m}, "
            f"cell_count={self._cell_count})"
        )
