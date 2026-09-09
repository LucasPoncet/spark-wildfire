"""Grid geometry the simulation runs on."""

from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class MeshConfiguration:
    """Extent, resolution and connectivity of the square grid.

    Attributes:
        extent_x_m: Domain extent along x, in metres.
        extent_y_m: Domain extent along y, in metres.
        cell_spacing_m: Distance between adjacent cell centres, in metres.
        use_diagonal_neighbors: 8-connectivity if True, 4-connectivity otherwise.
        mesh_type: Registered name of the mesh implementation.
        elevation_field_type: Registered name of the scalar field carrying
            ground height.
        elevation_m: Constant ground height, in metres, for a uniform field.
    """

    extent_x_m: float = 100.0
    extent_y_m: float = 100.0
    cell_spacing_m: float = 0.5
    use_diagonal_neighbors: bool = True
    mesh_type: str = "square_grid"
    elevation_field_type: str = "uniform"
    elevation_m: float = 0.0

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
            extent_x_m=float(data.get("extent_x_m", defaults.extent_x_m)),
            extent_y_m=float(data.get("extent_y_m", defaults.extent_y_m)),
            cell_spacing_m=float(data.get("cell_spacing_m", defaults.cell_spacing_m)),
            use_diagonal_neighbors=bool(
                data.get("use_diagonal_neighbors", defaults.use_diagonal_neighbors)
            ),
            mesh_type=str(data.get("mesh_type", defaults.mesh_type)),
            elevation_field_type=str(
                data.get("elevation_field_type", defaults.elevation_field_type)
            ),
            elevation_m=float(data.get("elevation_m", defaults.elevation_m)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "extent_x_m": self.extent_x_m,
            "extent_y_m": self.extent_y_m,
            "cell_spacing_m": self.cell_spacing_m,
            "use_diagonal_neighbors": self.use_diagonal_neighbors,
            "mesh_type": self.mesh_type,
            "elevation_field_type": self.elevation_field_type,
            "elevation_m": self.elevation_m,
        }
