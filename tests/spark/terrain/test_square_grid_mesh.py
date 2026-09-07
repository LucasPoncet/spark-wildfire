import numpy as np
import pytest

from spark.terrain.mesh_protocol import MeshProtocol
from spark.terrain.square_grid_mesh import SquareGridMesh, SquareGridMeshConfig

SMALL_EXTENT_M = 10.0
SMALL_SPACING_M = 1.0
SMALL_N_X = 11
LARGE_EXTENT_M = 100.0
LARGE_SPACING_M = 0.5


@pytest.fixture
def small_mesh_4_connectivity() -> SquareGridMesh:
    return SquareGridMesh.from_values(
        extent_x_m=SMALL_EXTENT_M,
        extent_y_m=SMALL_EXTENT_M,
        cell_spacing_m=SMALL_SPACING_M,
        use_diagonal_neighbors=False,
    )


@pytest.fixture
def small_mesh_8_connectivity() -> SquareGridMesh:
    return SquareGridMesh.from_values(
        extent_x_m=SMALL_EXTENT_M,
        extent_y_m=SMALL_EXTENT_M,
        cell_spacing_m=SMALL_SPACING_M,
        use_diagonal_neighbors=True,
    )


@pytest.fixture
def large_mesh_4_connectivity() -> SquareGridMesh:
    return SquareGridMesh(
        SquareGridMeshConfig(
            extent_x_m=LARGE_EXTENT_M,
            extent_y_m=LARGE_EXTENT_M,
            cell_spacing_m=LARGE_SPACING_M,
            use_diagonal_neighbors=False,
        )
    )


def test_cell_count_matches_expected_grid_dimensions(
    large_mesh_4_connectivity: SquareGridMesh,
) -> None:
    assert large_mesh_4_connectivity.n_x == 201
    assert large_mesh_4_connectivity.n_y == 201
    assert large_mesh_4_connectivity.cell_count == 40401
    assert large_mesh_4_connectivity.cell_positions_xyz.shape == (40401, 3)


def test_cell_positions_origin_is_zero(
    small_mesh_4_connectivity: SquareGridMesh,
) -> None:
    np.testing.assert_array_equal(
        small_mesh_4_connectivity.cell_positions_xyz[0], np.zeros(3)
    )


def test_cell_positions_corner_is_at_extent(
    large_mesh_4_connectivity: SquareGridMesh,
) -> None:
    np.testing.assert_allclose(
        large_mesh_4_connectivity.cell_positions_xyz[-1],
        np.array([LARGE_EXTENT_M, LARGE_EXTENT_M, 0.0]),
    )


def test_neighbor_indices_interior_cell_has_four_neighbors_in_4connectivity(
    small_mesh_4_connectivity: SquareGridMesh,
) -> None:
    center_cell = (SMALL_N_X // 2) * SMALL_N_X + SMALL_N_X // 2
    center_neighbors = small_mesh_4_connectivity.neighbor_indices[center_cell]
    assert small_mesh_4_connectivity.max_neighbors == 4
    assert center_neighbors.shape == (4,)
    assert not np.any(center_neighbors == -1)


def test_neighbor_indices_corner_cell_has_two_neighbors_in_4connectivity(
    small_mesh_4_connectivity: SquareGridMesh,
) -> None:
    corner_cells = [0, SMALL_N_X - 1, -SMALL_N_X, -1]
    corner_neighbors = small_mesh_4_connectivity.neighbor_indices[corner_cells]
    np.testing.assert_array_equal(np.sum(corner_neighbors >= 0, axis=1), 2)


def test_neighbor_unit_directions_are_unit_length_where_valid(
    small_mesh_8_connectivity: SquareGridMesh,
) -> None:
    is_valid = small_mesh_8_connectivity.neighbor_indices >= 0
    norms = np.linalg.norm(
        small_mesh_8_connectivity.neighbor_unit_directions_xyz, axis=2
    )
    np.testing.assert_allclose(norms[is_valid], 1.0)
    np.testing.assert_array_equal(norms[~is_valid], 0.0)


def test_neighbor_distances_diagonal_is_sqrt2_times_spacing(
    small_mesh_8_connectivity: SquareGridMesh,
) -> None:
    is_valid = small_mesh_8_connectivity.neighbor_indices >= 0
    displacements_xyz = (
        small_mesh_8_connectivity.cell_positions_xyz[
            np.where(is_valid, small_mesh_8_connectivity.neighbor_indices, 0)
        ]
        - small_mesh_8_connectivity.cell_positions_xyz[:, None, :]
    )
    is_diagonal = (
        is_valid
        & (displacements_xyz[:, :, 0] != 0.0)
        & (displacements_xyz[:, :, 1] != 0.0)
    )
    assert small_mesh_8_connectivity.max_neighbors == 8
    assert np.any(is_diagonal)
    np.testing.assert_allclose(
        small_mesh_8_connectivity.neighbor_distances_m[is_diagonal],
        SMALL_SPACING_M * np.sqrt(2.0),
    )
    np.testing.assert_allclose(
        small_mesh_8_connectivity.neighbor_distances_m[is_valid & ~is_diagonal],
        SMALL_SPACING_M,
    )


def test_satisfies_mesh_protocol(small_mesh_4_connectivity: SquareGridMesh) -> None:
    assert isinstance(small_mesh_4_connectivity, MeshProtocol)


def test_repr_contains_cell_count(small_mesh_4_connectivity: SquareGridMesh) -> None:
    assert repr(small_mesh_4_connectivity) == (
        "SquareGridMesh(n_x=11, n_y=11, cell_spacing_m=1.0, cell_count=121)"
    )
