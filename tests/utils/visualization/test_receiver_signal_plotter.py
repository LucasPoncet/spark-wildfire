import numpy as np
import pytest
from matplotlib.figure import Figure

from src.utils.visualization.receiver_signal_plotter import (
    plot_band_analysis_chain,
    plot_error_against_bisector_distance,
    plot_error_against_signal_to_noise_ratio,
    plot_estimates_against_truth,
    plot_front_position_over_observations,
    plot_receiver_level_traces,
    plot_reduced_chi_square_by_scenario,
)

OBSERVATION_TIMES_S = np.array([60.0, 120.0, 180.0, 240.0])
RECEIVER_POSITIONS_XY_M = np.array([[30.0, 0.0], [70.0, 0.0], [50.0, 60.0]])
BAND_CENTRES_HZ = np.array([125.0, 250.0, 500.0, 1000.0])


def build_receiver_levels_db() -> np.ndarray:
    return np.tile(
        np.array([-40.0, -42.0, -45.0]), (OBSERVATION_TIMES_S.size, 1)
    ) + np.arange(OBSERVATION_TIMES_S.size).reshape(-1, 1)


def test_level_traces_draw_one_line_per_receiver() -> None:
    figure = plot_receiver_level_traces(
        OBSERVATION_TIMES_S, build_receiver_levels_db(), RECEIVER_POSITIONS_XY_M
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes[0].get_lines()) == RECEIVER_POSITIONS_XY_M.shape[0]


def test_level_traces_reject_a_mismatched_level_array() -> None:
    with pytest.raises(ValueError, match="n_observations, n_receivers"):
        plot_receiver_level_traces(
            OBSERVATION_TIMES_S,
            build_receiver_levels_db()[:, :2],
            RECEIVER_POSITIONS_XY_M,
        )


def test_the_band_chain_has_three_panels() -> None:
    figure = plot_band_analysis_chain(
        BAND_CENTRES_HZ,
        np.array([-40.0, -41.0, -43.0, -47.0]),
        np.array([-42.0, -44.0, -47.0, -53.0]),
        np.array([0.0005, 0.0015, 0.004, 0.010]),
        13.0,
    )
    assert len(figure.axes) == 3


def test_estimates_against_truth_has_a_scene_and_a_zoom_panel() -> None:
    figure = plot_estimates_against_truth(
        np.array([[25.0, 70.0], [25.1, 69.9]]),
        np.array([2.6, 1.9]),
        np.array([0.003, 0.003]),
        np.array([25.0, 70.0]),
        RECEIVER_POSITIONS_XY_M,
        "scenario 1",
    )
    assert len(figure.axes) == 2
    assert "scene" in figure.axes[0].get_title()
    assert "zoom" in figure.axes[1].get_title()


def test_estimates_against_truth_draws_one_ellipse_per_clip_in_both_panels() -> None:
    figure = plot_estimates_against_truth(
        np.array([[25.0, 70.0], [25.1, 69.9], [24.9, 70.2]]),
        np.array([2.6, 1.9, 1.7]),
        np.array([0.003, 0.003, 0.003]),
        np.array([25.0, 70.0]),
        RECEIVER_POSITIONS_XY_M,
        "scenario 1",
    )
    assert len(figure.axes[0].patches) == 3
    assert len(figure.axes[1].patches) == 3


def test_the_zoom_panel_frames_the_estimate_spread() -> None:
    estimated_positions_xy_m = np.array([[25.0, 70.0], [25.1, 69.9], [24.9, 70.2]])
    true_position_xy_m = np.array([25.0, 70.0])
    figure = plot_estimates_against_truth(
        estimated_positions_xy_m,
        np.array([2.6, 1.9, 1.7]),
        np.array([0.003, 0.003, 0.003]),
        true_position_xy_m,
        RECEIVER_POSITIONS_XY_M,
        "scenario 1",
    )
    zoom_axes = figure.axes[1]
    left_m, right_m = zoom_axes.get_xlim()
    bottom_m, top_m = zoom_axes.get_ylim()
    assert right_m - left_m < 2.0
    assert left_m <= estimated_positions_xy_m[:, 0].min()
    assert right_m >= estimated_positions_xy_m[:, 0].max()
    assert bottom_m <= estimated_positions_xy_m[:, 1].min()
    assert top_m >= estimated_positions_xy_m[:, 1].max()


def test_the_zoom_panel_survives_a_perfect_estimate() -> None:
    true_position_xy_m = np.array([25.0, 70.0])
    figure = plot_estimates_against_truth(
        np.tile(true_position_xy_m, (3, 1)),
        np.array([0.1, 0.1, 0.1]),
        np.array([0.01, 0.01, 0.01]),
        true_position_xy_m,
        RECEIVER_POSITIONS_XY_M,
        "scenario 1",
    )
    left_m, right_m = figure.axes[1].get_xlim()
    assert right_m > left_m


def test_truth_stays_visible_when_the_estimates_cover_it() -> None:
    true_position_xy_m = np.array([25.0, 70.0])
    figure = plot_estimates_against_truth(
        np.tile(true_position_xy_m, (5, 1)),
        np.array([0.1] * 5),
        np.array([0.01] * 5),
        true_position_xy_m,
        RECEIVER_POSITIONS_XY_M,
        "scenario 1",
    )
    truth_labels = [
        collection
        for collection in figure.axes[0].collections
        if collection.get_label() == "truth"
    ]
    assert len(truth_labels) == 1
    assert truth_labels[0].get_zorder() > max(
        collection.get_zorder()
        for collection in figure.axes[0].collections
        if collection.get_label() == "estimates"
    )


def test_reduced_chi_square_marks_a_degenerate_scenario_instead_of_dropping_it() -> (
    None
):
    figure = plot_reduced_chi_square_by_scenario(
        ["S1", "S2", "S3"],
        [np.array([18.0, 30.0]), np.array([50.0, 78.0]), np.array([0.0, 0.0])],
        [False, False, True],
    )
    axes = figure.axes[0]
    assert [label.get_text() for label in axes.get_xticklabels()] == ["S1", "S2", "S3"]
    degenerate_labels = [
        collection.get_label()
        for collection in axes.collections
        if collection.get_label() == "degenerate, no fit"
    ]
    assert degenerate_labels == ["degenerate, no fit"]


def test_reduced_chi_square_still_draws_when_every_scenario_is_degenerate() -> None:
    figure = plot_reduced_chi_square_by_scenario(["S1"], [np.array([0.0, 0.0])], [True])
    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == ["S1"]


def test_reduced_chi_square_rejects_a_mismatched_degenerate_list() -> None:
    with pytest.raises(ValueError, match="exactly one label"):
        plot_reduced_chi_square_by_scenario(
            ["S1", "S2"],
            [np.array([1.0]), np.array([2.0])],
            [True],
        )


def test_error_against_noise_draws_one_line_per_scenario() -> None:
    figure = plot_error_against_signal_to_noise_ratio(
        np.array([10.0, 20.0, 30.0]),
        np.array([[1.0, 0.5, 0.2], [2.0, 1.0, 0.4]]),
        ["S1", "S2"],
    )
    assert len(figure.axes[0].get_lines()) == 2


def test_error_against_noise_rejects_a_mismatched_error_array() -> None:
    with pytest.raises(ValueError, match="n_scenarios, n_ratios"):
        plot_error_against_signal_to_noise_ratio(
            np.array([10.0, 20.0]), np.array([[1.0, 0.5, 0.2]]), ["S1"]
        )


def test_error_against_bisector_distance_shades_the_singular_band() -> None:
    figure = plot_error_against_bisector_distance(
        np.array([1.0, 5.0, 20.0]), np.array([3.0, 0.5, 0.02]), 2.0
    )
    assert len(figure.axes[0].patches) == 1


def test_error_against_bisector_distance_survives_a_zero_error() -> None:
    figure = plot_error_against_bisector_distance(
        np.array([1.0, 5.0]), np.array([0.0, 0.5]), 2.0
    )
    assert len(figure.axes) == 1


def test_reduced_chi_square_draws_one_box_per_scenario() -> None:
    figure = plot_reduced_chi_square_by_scenario(
        ["S1", "S2"], [np.array([18.0, 30.0, 44.0]), np.array([50.0, 78.0])]
    )
    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [
        "S1",
        "S2",
    ]


def test_reduced_chi_square_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="exactly one label"):
        plot_reduced_chi_square_by_scenario(["S1"], [np.array([1.0]), np.array([2.0])])


def test_front_position_has_two_panels_and_a_fit() -> None:
    figure = plot_front_position_over_observations(
        OBSERVATION_TIMES_S, np.array([1.0, 2.5, 4.0, 5.4]), None
    )
    assert len(figure.axes) == 2
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert any("mm/s" in text.get_text() for text in legend.texts)


def test_front_position_draws_the_estimate_when_given_one() -> None:
    figure = plot_front_position_over_observations(
        OBSERVATION_TIMES_S,
        np.array([1.0, 2.5, 4.0, 5.4]),
        np.array([1.1, 2.4, 4.2, 5.3]),
    )
    assert len(figure.axes[0].get_lines()) == 3
