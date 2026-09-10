import numpy as np
import pytest
from matplotlib.figure import Figure

from src.utils.visualization.mesh_plotter import (
    compute_near_singular_band_mask,
    compute_perpendicular_bisector_endpoints_xy_m,
    compute_range_ratio_field,
    plot_scene_geometry,
    plot_scene_geometry_panels,
)

EXTENT_M = 100.0
NEAR_SINGULAR_TOLERANCE = 0.05
RECEIVER_1_XY_M = np.array([30.0, 0.0])
RECEIVER_2_XY_M = np.array([70.0, 0.0])
RECEIVERS_XY_M = np.stack((RECEIVER_1_XY_M, RECEIVER_2_XY_M))
IGNITION_XY_M = np.array([[30.0, 30.0]])


def test_range_ratio_is_one_on_the_bisector() -> None:
    on_bisector_xy_m = np.array([[50.0, 10.0], [50.0, 40.0], [50.0, 90.0]])
    np.testing.assert_allclose(
        compute_range_ratio_field(on_bisector_xy_m, RECEIVER_1_XY_M, RECEIVER_2_XY_M),
        1.0,
    )


def test_the_band_covers_the_bisector_and_not_the_receivers() -> None:
    sample_positions_xy_m = np.array([[50.0, 40.0], [31.0, 5.0]])
    mask = compute_near_singular_band_mask(
        sample_positions_xy_m,
        RECEIVER_1_XY_M,
        RECEIVER_2_XY_M,
        NEAR_SINGULAR_TOLERANCE,
    )
    assert bool(mask[0])
    assert not bool(mask[1])


def test_a_wider_tolerance_never_shrinks_the_band() -> None:
    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(0.0, EXTENT_M, 40), np.linspace(0.0, EXTENT_M, 40)
    )
    samples_xy_m = np.stack((grid_x_m.ravel(), grid_y_m.ravel()), axis=1)
    narrow = compute_near_singular_band_mask(
        samples_xy_m, RECEIVER_1_XY_M, RECEIVER_2_XY_M, 0.02
    )
    wide = compute_near_singular_band_mask(
        samples_xy_m, RECEIVER_1_XY_M, RECEIVER_2_XY_M, 0.20
    )
    assert bool(np.all(wide >= narrow))
    assert int(wide.sum()) > int(narrow.sum())


def test_the_bisector_passes_through_the_midpoint() -> None:
    endpoints_xy_m = compute_perpendicular_bisector_endpoints_xy_m(
        RECEIVER_1_XY_M, RECEIVER_2_XY_M, EXTENT_M, EXTENT_M
    )
    np.testing.assert_allclose(endpoints_xy_m.mean(axis=0), [50.0, 0.0], atol=1e-9)


def test_the_bisector_is_perpendicular_to_the_baseline() -> None:
    endpoints_xy_m = compute_perpendicular_bisector_endpoints_xy_m(
        RECEIVER_1_XY_M, RECEIVER_2_XY_M, EXTENT_M, EXTENT_M
    )
    direction = endpoints_xy_m[1] - endpoints_xy_m[0]
    baseline = RECEIVER_2_XY_M - RECEIVER_1_XY_M
    assert float(np.dot(direction, baseline)) == pytest.approx(0.0, abs=1e-9)


def test_coincident_receivers_have_no_bisector() -> None:
    with pytest.raises(ValueError, match="different positions"):
        compute_perpendicular_bisector_endpoints_xy_m(
            RECEIVER_1_XY_M, RECEIVER_1_XY_M, EXTENT_M, EXTENT_M
        )


def test_scene_geometry_returns_one_axes() -> None:
    figure = plot_scene_geometry(
        EXTENT_M,
        EXTENT_M,
        IGNITION_XY_M,
        RECEIVERS_XY_M,
        NEAR_SINGULAR_TOLERANCE,
        "E3",
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1


def test_scene_geometry_survives_a_single_receiver() -> None:
    figure = plot_scene_geometry(
        EXTENT_M,
        EXTENT_M,
        IGNITION_XY_M,
        RECEIVER_1_XY_M.reshape(1, 2),
        NEAR_SINGULAR_TOLERANCE,
        "one receiver",
    )
    assert len(figure.axes) == 1


def test_the_panel_figure_has_one_axes_per_scene() -> None:
    figure = plot_scene_geometry_panels(
        ["E1", "E2", "E3"],
        [(EXTENT_M, EXTENT_M)] * 3,
        [IGNITION_XY_M] * 3,
        [RECEIVERS_XY_M] * 3,
        NEAR_SINGULAR_TOLERANCE,
    )
    assert len(figure.axes) == 3


def test_the_panel_figure_rejects_mismatched_lists() -> None:
    with pytest.raises(ValueError, match="every scene must supply"):
        plot_scene_geometry_panels(
            ["E1", "E2"],
            [(EXTENT_M, EXTENT_M)],
            [IGNITION_XY_M],
            [RECEIVERS_XY_M],
            NEAR_SINGULAR_TOLERANCE,
        )
