import numpy as np
import pytest
from matplotlib.figure import Figure

from src.utils.visualization.fire_characterization_plotter import (
    plot_fire_characterization,
)

TIMES_S = np.arange(5.0, 55.0, 5.0)
TRUE_BEARINGS_DEG = np.zeros(TIMES_S.size)
ESTIMATED_BEARINGS_DEG = np.array(
    [12.0, -6.0, 0.4, 1.1, 0.3, -2.2, -1.0, 0.8, 0.2, -0.4]
)
FUSED_BEARING_DEG = -0.6
TRUE_SEMI_AXES_M = np.array(
    [0.45, 0.76, 1.31, 1.68, 2.40, 2.90, 3.42, 4.12, 4.54, 5.33]
)
ESTIMATED_SEMI_AXES_M = np.array(
    [2.12, 2.20, 0.79, 0.96, 1.60, 2.02, 2.44, 3.11, 3.60, 4.30]
)
IS_RESOLVED = np.array([False, False, True, True, True, True, True, True, True, True])
RESPONSE_SEMI_AXES_M = np.full(TIMES_S.size, 1.55)


def build_figure() -> Figure:
    return plot_fire_characterization(
        TIMES_S,
        ESTIMATED_BEARINGS_DEG,
        TRUE_BEARINGS_DEG,
        FUSED_BEARING_DEG,
        ESTIMATED_SEMI_AXES_M,
        TRUE_SEMI_AXES_M,
        IS_RESOLVED,
        RESPONSE_SEMI_AXES_M,
    )


def test_the_figure_has_a_bearing_panel_and_an_extent_panel() -> None:
    figure = build_figure()
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "direction of spread"
    assert figure.axes[1].get_title() == "front extent"


def test_the_bearing_panel_draws_truth_estimate_and_the_fused_answer() -> None:
    bearing_axes = build_figure().axes[0]
    assert len(bearing_axes.lines) == 3
    labels = [line.get_label() for line in bearing_axes.lines]
    assert "true" in labels
    assert any("fused" in str(label) for label in labels)


def test_the_extent_panel_shows_the_response_it_had_to_beat() -> None:
    extent_axes = build_figure().axes[1]
    labels = [str(line.get_label()) for line in extent_axes.lines]
    assert "array response" in labels


def test_resolved_and_unresolved_frames_are_drawn_apart() -> None:
    extent_axes = build_figure().axes[1]
    labels = [str(collection.get_label()) for collection in extent_axes.collections]
    assert "resolved" in labels
    assert "upper bound only" in labels


def test_a_run_with_every_frame_resolved_still_draws() -> None:
    figure = plot_fire_characterization(
        TIMES_S,
        ESTIMATED_BEARINGS_DEG,
        TRUE_BEARINGS_DEG,
        FUSED_BEARING_DEG,
        ESTIMATED_SEMI_AXES_M,
        TRUE_SEMI_AXES_M,
        np.ones(TIMES_S.size, dtype=bool),
        RESPONSE_SEMI_AXES_M,
    )
    assert isinstance(figure, Figure)


def test_a_run_with_nothing_resolved_still_draws() -> None:
    figure = plot_fire_characterization(
        TIMES_S,
        ESTIMATED_BEARINGS_DEG,
        TRUE_BEARINGS_DEG,
        FUSED_BEARING_DEG,
        ESTIMATED_SEMI_AXES_M,
        TRUE_SEMI_AXES_M,
        np.zeros(TIMES_S.size, dtype=bool),
        RESPONSE_SEMI_AXES_M,
    )
    assert isinstance(figure, Figure)


def test_series_of_different_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_fire_characterization(
            TIMES_S[:3],
            ESTIMATED_BEARINGS_DEG,
            TRUE_BEARINGS_DEG,
            FUSED_BEARING_DEG,
            ESTIMATED_SEMI_AXES_M,
            TRUE_SEMI_AXES_M,
            IS_RESOLVED,
            RESPONSE_SEMI_AXES_M,
        )


def test_both_panels_run_on_the_same_clock() -> None:
    figure = build_figure()
    assert figure.axes[0].get_xlabel() == "simulated time (s)"
    assert figure.axes[1].get_xlabel() == "simulated time (s)"
