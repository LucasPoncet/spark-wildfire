import numpy as np
import pytest
from matplotlib.figure import Figure

from src.utils.visualization.map_linearity_plotter import (
    build_series_label,
    plot_map_linearity_audit,
)

SOURCE_COUNTS = [2, 2, 2, 2, 8, 8, 8, 8]
EXPONENTS = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
COMBINATORS = ["sum", "sum", "product", "product"] * 2
POOLINGS = ["max"] * 8
RESIDUALS = [0.04, 0.05, 0.41, 0.82, 0.05, 0.05, 0.42, 0.83]
CORRELATIONS = [0.999, 0.998, 0.93, 0.90, 0.997, 0.997, 0.92, 0.90]
RESIDUAL_TARGET = 0.05


def build_figure() -> Figure:
    return plot_map_linearity_audit(
        SOURCE_COUNTS,
        EXPONENTS,
        COMBINATORS,
        POOLINGS,
        RESIDUALS,
        CORRELATIONS,
        RESIDUAL_TARGET,
    )


def test_a_series_label_names_the_combinator_and_the_pooling() -> None:
    assert build_series_label("sum", "max") == "sum, max"


def test_there_is_one_panel_per_source_count() -> None:
    figure = build_figure()
    assert isinstance(figure, Figure)
    assert len(figure.axes) == len(set(SOURCE_COUNTS))


def test_each_panel_is_titled_with_its_source_count() -> None:
    titles = [axes.get_title() for axes in build_figure().axes]
    assert titles == ["2 sources", "8 sources"]


def test_each_panel_draws_one_line_per_series_plus_the_target_rule() -> None:
    series_count = len(set(zip(COMBINATORS, POOLINGS, strict=True)))
    for axes in build_figure().axes:
        assert len(axes.lines) == series_count + 1


def test_the_residual_axis_is_logarithmic_so_a_decade_is_visible() -> None:
    assert build_figure().axes[0].get_yscale() == "log"


def test_a_single_source_count_still_draws_one_panel() -> None:
    figure = plot_map_linearity_audit(
        [8, 8],
        [0.0, 1.0],
        ["sum", "sum"],
        ["max", "max"],
        [0.05, 0.06],
        [0.99, 0.98],
        0.05,
    )
    assert len(figure.axes) == 1


def test_columns_of_different_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_map_linearity_audit(
            [2], [0.0, 1.0], ["sum"], ["max"], [0.05], [0.99], 0.05
        )


def test_the_figure_survives_a_series_that_is_missing_from_one_panel() -> None:
    figure = plot_map_linearity_audit(
        [2, 8, 8],
        [0.0, 0.0, 1.0],
        ["sum", "product", "product"],
        ["max", "max", "max"],
        [0.04, 0.42, 0.83],
        [0.999, 0.92, 0.90],
        0.05,
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2


def test_the_residuals_reach_the_axis_in_ascending_exponent_order() -> None:
    figure = plot_map_linearity_audit(
        [8, 8, 8],
        [1.0, 0.0, 0.3],
        ["sum", "sum", "sum"],
        ["max", "max", "max"],
        [0.054, 0.049, 0.050],
        [0.997, 0.997, 0.997],
        0.05,
    )
    line = figure.axes[0].lines[0]
    assert np.asarray(line.get_xdata()) == pytest.approx([0.0, 0.3, 1.0])
    assert np.asarray(line.get_ydata()) == pytest.approx([0.049, 0.050, 0.054])
