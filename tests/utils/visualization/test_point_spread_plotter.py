import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.utils.visualization.point_spread_plotter import (
    plot_point_spread_calibration,
)

NODE_SPACING_M: float = 10.0
DOMAIN_EXTENT_M: float = 60.0
RECEIVERS_XYZ_M = np.array(
    [[5.0, 30.0, 1.5], [55.0, 30.0, 1.5], [30.0, 55.0, 1.5], [30.0, 5.0, 1.5]]
)
VALIDATION_POSITIONS_XY_M = np.array([[30.0, 30.0], [42.0, 30.0], [51.0, 30.0]])
VALIDATION_CORRELATIONS = np.array([0.99, 0.97, 0.95])


def build_nodes() -> tuple[np.ndarray, tuple[int, int]]:
    axis_m = np.arange(5.0, DOMAIN_EXTENT_M, NODE_SPACING_M)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    nodes_xy_m = np.stack((grid_x_m.ravel(), grid_y_m.ravel()), axis=1)
    return nodes_xy_m, (axis_m.size, axis_m.size)


def build_figure() -> Figure:
    nodes_xy_m, grid_shape = build_nodes()
    centre_xy_m = np.array([0.5 * DOMAIN_EXTENT_M, 0.5 * DOMAIN_EXTENT_M])
    offsets_xy_m = 0.02 * (nodes_xy_m - centre_xy_m)
    semi_axes_m = 6.0 + 0.05 * np.linalg.norm(nodes_xy_m - centre_xy_m, axis=1)
    return plot_point_spread_calibration(
        nodes_xy_m,
        offsets_xy_m,
        semi_axes_m,
        grid_shape,
        RECEIVERS_XYZ_M,
        VALIDATION_CORRELATIONS,
        VALIDATION_POSITIONS_XY_M,
    )


def find_panel(figure: Figure, title: str) -> Axes:
    for axes in figure.axes:
        if axes.get_title() == title or axes.get_title().startswith(title):
            return axes
    raise AssertionError(f"no panel titled {title!r}")


def test_the_calibration_figure_has_one_panel_for_each_field_and_the_check() -> None:
    figure = build_figure()
    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 3


def test_the_bias_panel_reports_the_largest_offset_in_its_title() -> None:
    assert "largest" in find_panel(build_figure(), "centroid bias").get_title()


def test_the_bias_and_width_panels_keep_the_domain_square() -> None:
    figure = build_figure()
    assert find_panel(figure, "centroid bias").get_aspect() == 1.0
    assert find_panel(figure, "response width").get_aspect() == 1.0


def test_the_validation_panel_draws_one_bar_per_checked_position() -> None:
    validation_axes = find_panel(build_figure(), "response validation")
    assert len(validation_axes.patches) == VALIDATION_POSITIONS_XY_M.shape[0]


def test_the_validation_panel_is_scaled_so_a_correlation_is_readable() -> None:
    validation_axes = find_panel(build_figure(), "response validation")
    assert validation_axes.get_ylim() == pytest.approx((0.0, 1.05))


def test_the_width_panel_carries_a_colour_bar_naming_its_units() -> None:
    labels = [axes.get_ylabel() for axes in build_figure().axes]
    assert "major semi-axis (m)" in labels


def test_a_node_grid_that_does_not_match_its_shape_is_rejected() -> None:
    nodes_xy_m, _ = build_nodes()
    with pytest.raises(ValueError, match="does not match the number of nodes"):
        plot_point_spread_calibration(
            nodes_xy_m,
            np.zeros_like(nodes_xy_m),
            np.ones(nodes_xy_m.shape[0]),
            (2, 2),
            RECEIVERS_XYZ_M,
            VALIDATION_CORRELATIONS,
            VALIDATION_POSITIONS_XY_M,
        )
