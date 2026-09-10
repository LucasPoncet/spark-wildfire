import numpy as np
import pytest
from matplotlib.figure import Figure

from src.utils.visualization.localization_error_plotter import (
    plot_metric_against_parameter,
    plot_source_count_confusion,
)


def test_one_line_is_drawn_per_series() -> None:
    figure = plot_metric_against_parameter(
        np.array([2.0, 10.0, 40.0]),
        np.array([[8.0, 2.0, 0.5], [12.0, 4.0, 1.0]]),
        ["product", "sum"],
        "source separation (m)",
        "optimal sub-pattern assignment (m)",
        "resolution against separation",
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes[0].lines) == 2
    assert figure.axes[0].get_xlabel() == "source separation (m)"


def test_a_single_series_needs_no_legend() -> None:
    figure = plot_metric_against_parameter(
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0]]),
        ["only"],
        "x",
        "y",
        "one series",
    )
    assert figure.axes[0].get_legend() is None


def test_a_label_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="one label is required"):
        plot_metric_against_parameter(
            np.array([1.0, 2.0]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            ["only"],
            "x",
            "y",
            "mismatch",
        )


def test_the_confusion_figure_draws_the_diagonal() -> None:
    figure = plot_source_count_confusion(
        np.array([1.0, 2.0, 3.0, 3.0]),
        np.array([1.0, 2.0, 2.0, 4.0]),
        "estimated against true source count",
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes[0].lines) == 1
    assert figure.axes[0].get_xlim() == figure.axes[0].get_ylim()
