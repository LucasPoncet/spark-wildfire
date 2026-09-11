import numpy as np
import pytest
from matplotlib.figure import Figure

from src.utils.visualization.speckle_floor_plotter import plot_speckle_floor

FRAME_DURATIONS_S = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
SPECKLE_FLOORS = np.array([0.42, 0.31, 0.22, 0.16, 0.11])
CONFIGURED_FRAME_DURATION_S = 4.0


def build_figure() -> Figure:
    return plot_speckle_floor(
        FRAME_DURATIONS_S, SPECKLE_FLOORS, CONFIGURED_FRAME_DURATION_S
    )


def test_the_curve_carries_every_measured_point() -> None:
    figure = build_figure()
    assert isinstance(figure, Figure)
    curve = figure.axes[0].lines[0]
    assert np.asarray(curve.get_xdata()) == pytest.approx(FRAME_DURATIONS_S)
    assert np.asarray(curve.get_ydata()) == pytest.approx(SPECKLE_FLOORS)


def test_both_axes_are_logarithmic_so_a_power_law_reads_straight() -> None:
    axes = build_figure().axes[0]
    assert axes.get_xscale() == "log"
    assert axes.get_yscale() == "log"


def test_the_configured_duration_is_marked_on_the_curve() -> None:
    axes = build_figure().axes[0]
    marker = axes.lines[1]
    assert np.asarray(marker.get_xdata()) == pytest.approx(
        [CONFIGURED_FRAME_DURATION_S, CONFIGURED_FRAME_DURATION_S]
    )


def test_the_marker_says_which_duration_it_is() -> None:
    assert "4.0 s" in str(build_figure().axes[0].lines[1].get_label())


def test_mismatched_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one speckle floor"):
        plot_speckle_floor(FRAME_DURATIONS_S, SPECKLE_FLOORS[:3], 4.0)
