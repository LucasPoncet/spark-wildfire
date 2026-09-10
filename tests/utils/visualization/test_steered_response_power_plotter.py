import numpy as np
import pytest
from matplotlib.figure import Figure

from src.spark.inverse.steered_response_power import build_candidate_grid_xyz
from src.utils.visualization.steered_response_power_plotter import (
    plot_steered_response_power_map,
    plot_windowed_position_clusters,
    reshape_map_to_grid,
)

DOMAIN_EXTENT_M: float = 40.0
SPACING_M: float = 4.0
SOURCE_HEIGHT_M: float = 0.5
RECEIVERS_XYZ_M = np.array(
    [[5.0, 20.0, 1.5], [35.0, 20.0, 1.5], [20.0, 35.0, 1.5], [20.0, 5.0, 1.5]]
)


def build_grid_and_map() -> tuple[np.ndarray, np.ndarray]:
    grid = build_candidate_grid_xyz(
        DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, SPACING_M, SOURCE_HEIGHT_M
    )
    distances_m = np.linalg.norm(
        grid.positions_xyz_m[:, :2] - np.array([14.0, 26.0]), axis=1
    )
    return grid.positions_xyz_m, np.exp(-0.5 * (distances_m / 4.0) ** 2)


def test_a_flat_map_is_folded_back_into_its_grid() -> None:
    positions_xyz_m, power_map = build_grid_and_map()
    x_m, y_m, map_grid = reshape_map_to_grid(power_map, positions_xyz_m)
    assert map_grid.shape == (y_m.size, x_m.size)
    assert map_grid.size == power_map.size


def test_a_non_rectangular_candidate_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="rectangular grid"):
        reshape_map_to_grid(
            np.zeros(3), np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 0.5], [2.0, 3.0, 0.5]])
        )


def test_the_map_figure_carries_receivers_truth_and_estimates() -> None:
    positions_xyz_m, power_map = build_grid_and_map()
    figure = plot_steered_response_power_map(
        power_map,
        positions_xyz_m,
        RECEIVERS_XYZ_M,
        np.array([[14.0, 26.0]]),
        np.array([[14.5, 25.5]]),
        [np.diag([4.0, 1.0])],
        "test map",
    )
    assert isinstance(figure, Figure)
    assert figure.axes[0].get_title() == "test map"
    assert len(figure.axes[0].collections) >= 3


def test_the_map_figure_survives_having_found_nothing() -> None:
    positions_xyz_m, power_map = build_grid_and_map()
    figure = plot_steered_response_power_map(
        power_map,
        positions_xyz_m,
        RECEIVERS_XYZ_M,
        np.array([[14.0, 26.0]]),
        np.empty((0, 2)),
        [],
        "empty",
    )
    assert isinstance(figure, Figure)


def test_the_cluster_figure_draws_the_window_maxima() -> None:
    generator = np.random.default_rng(0)
    figure = plot_windowed_position_clusters(
        np.array([14.0, 26.0]) + generator.standard_normal((30, 2)),
        np.array([[14.0, 26.0]]),
        np.array([[14.0, 26.0]]),
        RECEIVERS_XYZ_M,
    )
    assert isinstance(figure, Figure)
    assert figure.axes[0].get_xlabel() == "x (m)"


def test_the_cluster_figure_survives_having_found_no_cluster() -> None:
    figure = plot_windowed_position_clusters(
        np.random.default_rng(1).standard_normal((10, 2)),
        np.empty((0, 2)),
        np.empty((0, 2)),
        RECEIVERS_XYZ_M,
    )
    assert isinstance(figure, Figure)
