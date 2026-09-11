"""The audit's own arithmetic, separately from the measurement it performs.

The measurement needs renders and is what the script exists to do. What is
checked here is the part that would be silently wrong: a comparison that
rewards the wrong family, or a selection rule that reads the sweep at a source
count the gate does not care about.
"""

import numpy as np
import pytest

from scripts.run_map_linearity_audit import (
    GATE_SOURCE_COUNT,
    LinearityCell,
    build_source_positions_xyz_m,
    compare_maps,
    select_imaging_configuration,
)

CENTRE_XY_M = np.array([30.0, 30.0])
RING_RADIUS_M: float = 5.0
SOURCE_HEIGHT_M: float = 0.5


def build_cell(
    source_count: int, exponent: float, combinator: str, pooling: str, residual: float
) -> LinearityCell:
    return LinearityCell(
        source_count=source_count,
        phase_transform_exponent=exponent,
        pairwise_combinator=combinator,
        pooling=pooling,
        correlation=1.0 - residual,
        normalized_residual=residual,
        grid_spacing_m=2.0,
    )


def test_sources_are_placed_on_a_ring_of_the_requested_radius() -> None:
    positions_xyz_m = build_source_positions_xyz_m(
        8, CENTRE_XY_M, RING_RADIUS_M, SOURCE_HEIGHT_M
    )
    radii_m = np.linalg.norm(positions_xyz_m[:, :2] - CENTRE_XY_M, axis=1)
    assert positions_xyz_m.shape == (8, 3)
    assert radii_m == pytest.approx(RING_RADIUS_M)


def test_every_source_sits_at_the_configured_height() -> None:
    positions_xyz_m = build_source_positions_xyz_m(
        6, CENTRE_XY_M, RING_RADIUS_M, SOURCE_HEIGHT_M
    )
    assert positions_xyz_m[:, 2] == pytest.approx(SOURCE_HEIGHT_M)


def test_sources_are_evenly_spaced_around_the_ring() -> None:
    positions_xyz_m = build_source_positions_xyz_m(
        6, CENTRE_XY_M, RING_RADIUS_M, SOURCE_HEIGHT_M
    )
    separations_m = np.linalg.norm(
        np.diff(positions_xyz_m[:, :2], axis=0, append=positions_xyz_m[:1, :2]), axis=1
    )
    assert separations_m == pytest.approx(separations_m[0])


def test_an_exactly_additive_pair_scores_no_residual() -> None:
    generator = np.random.default_rng(0)
    joint = generator.random(256)
    correlation, residual = compare_maps(joint, 2.0 * joint)
    assert correlation == pytest.approx(1.0)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_the_comparison_ignores_a_difference_of_overall_scale() -> None:
    generator = np.random.default_rng(1)
    joint = generator.random(256)
    _, residual = compare_maps(joint, 1000.0 * joint)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_a_map_that_disagrees_scores_a_residual() -> None:
    generator = np.random.default_rng(2)
    joint = generator.random(256)
    summed = generator.random(256)
    _, residual = compare_maps(joint, summed)
    assert residual > 0.1


def test_an_empty_joint_map_is_reported_as_infinitely_far_off() -> None:
    correlation, residual = compare_maps(np.zeros(64), np.ones(64))
    assert correlation == 0.0
    assert residual == float("inf")


def test_negative_cells_are_clipped_before_the_comparison() -> None:
    generator = np.random.default_rng(3)
    joint = generator.random(256)
    _, residual = compare_maps(joint, np.where(joint > 0.5, joint, -1.0))
    assert np.isfinite(residual)


def test_the_selection_reads_the_sweep_at_the_gate_source_count() -> None:
    cells = [
        build_cell(2, 0.0, "sum", "mean", 0.001),
        build_cell(GATE_SOURCE_COUNT, 0.0, "sum", "max", 0.05),
        build_cell(GATE_SOURCE_COUNT, 1.0, "product", "mean", 0.65),
    ]
    chosen = select_imaging_configuration(cells)
    assert chosen.source_count == GATE_SOURCE_COUNT
    assert chosen.pairwise_combinator == "sum"
    assert chosen.pooling == "max"


def test_the_selection_takes_the_smallest_residual_not_the_best_correlation() -> None:
    cells = [
        build_cell(GATE_SOURCE_COUNT, 0.0, "sum", "max", 0.049),
        build_cell(GATE_SOURCE_COUNT, 0.0, "sum", "mean", 0.061),
    ]
    assert select_imaging_configuration(cells).pooling == "max"


def test_the_selection_falls_back_to_the_largest_count_measured() -> None:
    cells = [
        build_cell(2, 0.0, "sum", "max", 0.30),
        build_cell(4, 0.0, "sum", "max", 0.04),
    ]
    assert select_imaging_configuration(cells).source_count == 4


def test_selecting_from_an_empty_sweep_is_rejected() -> None:
    with pytest.raises(ValueError, match="measured no combination"):
        select_imaging_configuration([])


def test_a_cell_serialises_every_field_it_carries() -> None:
    record = build_cell(8, 0.0, "sum", "max", 0.049).to_dict()
    assert set(record) == {
        "source_count",
        "phase_transform_exponent",
        "pairwise_combinator",
        "pooling",
        "correlation",
        "normalized_residual",
        "grid_spacing_m",
    }
