"""Interpolating and undoing the response's centroid offset.

The correction is by subtraction and iterated, because the offset is a
function of where the source is rather than of where its centroid landed. A
single subtraction leaves the residual the offset's own gradient produces, so
the round trip is what has to be asserted, not the lookup alone.
"""

from pathlib import Path

import numpy as np
import pytest

from src.spark.inverse.centroid_bias_correction import (
    CentroidBiasField,
    apply_centroid_bias_correction,
    compute_maximum_offset_magnitude_m,
    has_no_sign_flip_between_adjacent_nodes,
    interpolate_bias_offset_xy_m,
    load_centroid_bias_field,
    save_centroid_bias_field,
)

NODE_SPACING_M: float = 5.0
DOMAIN_EXTENT_M: float = 60.0
RADIAL_BIAS_FRACTION: float = 0.02


def build_radial_field() -> CentroidBiasField:
    axis_m = np.arange(2.5, DOMAIN_EXTENT_M, NODE_SPACING_M)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    nodes_xy_m = np.stack((grid_x_m.ravel(), grid_y_m.ravel()), axis=1)
    centre_xy_m = np.array([0.5 * DOMAIN_EXTENT_M, 0.5 * DOMAIN_EXTENT_M])
    return CentroidBiasField(
        node_positions_xy_m=nodes_xy_m,
        offsets_xy_m=RADIAL_BIAS_FRACTION * (nodes_xy_m - centre_xy_m),
        grid_shape=(axis_m.size, axis_m.size),
    )


def build_uniform_field(offset_xy_m: np.ndarray) -> CentroidBiasField:
    axis_m = np.arange(2.5, DOMAIN_EXTENT_M, NODE_SPACING_M)
    grid_x_m, grid_y_m = np.meshgrid(axis_m, axis_m)
    nodes_xy_m = np.stack((grid_x_m.ravel(), grid_y_m.ravel()), axis=1)
    return CentroidBiasField(
        node_positions_xy_m=nodes_xy_m,
        offsets_xy_m=np.tile(offset_xy_m, (nodes_xy_m.shape[0], 1)),
        grid_shape=(axis_m.size, axis_m.size),
    )


def test_interpolating_at_a_node_returns_that_node_offset() -> None:
    field = build_radial_field()
    node_index = 37
    assert interpolate_bias_offset_xy_m(
        field, field.node_positions_xy_m[node_index]
    ) == pytest.approx(field.offsets_xy_m[node_index])


def test_interpolating_between_two_nodes_lands_between_their_offsets() -> None:
    field = build_radial_field()
    midpoint_xy_m = 0.5 * (field.node_positions_xy_m[0] + field.node_positions_xy_m[1])
    interpolated = interpolate_bias_offset_xy_m(field, midpoint_xy_m)
    assert interpolated == pytest.approx(
        0.5 * (field.offsets_xy_m[0] + field.offsets_xy_m[1])
    )


def test_a_linear_field_is_reproduced_exactly_between_nodes() -> None:
    field = build_radial_field()
    query_xy_m = np.array([18.3, 41.7])
    centre_xy_m = np.array([0.5 * DOMAIN_EXTENT_M, 0.5 * DOMAIN_EXTENT_M])
    assert interpolate_bias_offset_xy_m(field, query_xy_m) == pytest.approx(
        RADIAL_BIAS_FRACTION * (query_xy_m - centre_xy_m), abs=1e-9
    )


def test_a_position_outside_the_table_is_clamped_rather_than_extrapolated() -> None:
    field = build_radial_field()
    far_outside = interpolate_bias_offset_xy_m(field, np.array([500.0, 500.0]))
    at_the_corner = interpolate_bias_offset_xy_m(field, field.node_positions_xy_m[-1])
    assert far_outside == pytest.approx(at_the_corner)


def test_a_uniform_bias_is_removed_exactly() -> None:
    offset_xy_m = np.array([0.4, -0.25])
    field = build_uniform_field(offset_xy_m)
    true_xy_m = np.array([22.0, 34.0])
    corrected = apply_centroid_bias_correction(true_xy_m + offset_xy_m, field)
    assert corrected == pytest.approx(true_xy_m)


def test_a_position_dependent_bias_is_removed_to_its_own_second_order() -> None:
    field = build_radial_field()
    centre_xy_m = np.array([0.5 * DOMAIN_EXTENT_M, 0.5 * DOMAIN_EXTENT_M])
    true_xy_m = np.array([44.0, 20.0])
    observed_xy_m = true_xy_m + RADIAL_BIAS_FRACTION * (true_xy_m - centre_xy_m)
    corrected = apply_centroid_bias_correction(observed_xy_m, field)
    assert float(np.linalg.norm(corrected - true_xy_m)) < 0.02 * float(
        np.linalg.norm(observed_xy_m - true_xy_m)
    )


def test_iterating_beats_a_single_subtraction_on_a_varying_field() -> None:
    field = build_radial_field()
    centre_xy_m = np.array([0.5 * DOMAIN_EXTENT_M, 0.5 * DOMAIN_EXTENT_M])
    true_xy_m = np.array([50.0, 12.0])
    observed_xy_m = true_xy_m + RADIAL_BIAS_FRACTION * (true_xy_m - centre_xy_m)
    once = apply_centroid_bias_correction(observed_xy_m, field, iteration_count=1)
    twice = apply_centroid_bias_correction(observed_xy_m, field, iteration_count=2)
    assert float(np.linalg.norm(twice - true_xy_m)) < float(
        np.linalg.norm(once - true_xy_m)
    )


def test_asking_for_no_iteration_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one iteration"):
        apply_centroid_bias_correction(np.zeros(2), build_radial_field(), 0)


def test_a_radial_field_is_reported_as_free_of_sign_flips() -> None:
    assert has_no_sign_flip_between_adjacent_nodes(build_radial_field())


def test_a_field_that_alternates_direction_is_reported_as_noisy() -> None:
    field = build_radial_field()
    alternating = np.asarray(field.offsets_xy_m).copy()
    alternating[::2] *= -1.0
    noisy = CentroidBiasField(
        node_positions_xy_m=field.node_positions_xy_m,
        offsets_xy_m=alternating,
        grid_shape=field.grid_shape,
    )
    assert not has_no_sign_flip_between_adjacent_nodes(noisy)


def test_the_largest_offset_is_reported() -> None:
    field = build_uniform_field(np.array([0.3, 0.4]))
    assert compute_maximum_offset_magnitude_m(field) == pytest.approx(0.5)


def test_a_saved_field_reloads_unchanged(tmp_path: Path) -> None:
    field = build_radial_field()
    reloaded = load_centroid_bias_field(
        save_centroid_bias_field(tmp_path / "bias.npz", field)
    )
    assert reloaded.node_positions_xy_m == pytest.approx(field.node_positions_xy_m)
    assert reloaded.offsets_xy_m == pytest.approx(field.offsets_xy_m)
    assert reloaded.grid_shape == field.grid_shape


def test_loading_a_field_that_is_not_there_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="centroid bias field not found"):
        load_centroid_bias_field(tmp_path / "absent.npz")
