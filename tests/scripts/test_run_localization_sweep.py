"""The sweep's grid expansion and figure reduction.

`scripts/` holds no domain logic, but it does hold the arithmetic that turns a
command line into cells and cells into curves. Both are silent when wrong: a
mis-expanded grid runs the wrong experiment and a mis-reduced series draws a
figure that looks fine.
"""

import argparse
import string

import numpy as np
import pytest

from scripts.run_localization_sweep import (
    SweepCell,
    SweepOutcome,
    build_sweep_grid,
    describe_cell,
    place_two_sources_at_separation,
    select_scene_sources,
    summarise_over,
)
from src.config.simulation_configuration import (
    SimulationConfiguration,
    load_simulation_configuration,
)
from src.config.source_scene_configuration import SourceConfiguration
from tests.conftest import CONFIGURATION_DIRECTORY

DOMAIN_EXTENT_M: float = 100.0


def build_arguments(**overrides: object) -> argparse.Namespace:
    defaults = {
        "receiver_counts": [4],
        "signal_to_noise_ratios_db": [20.0],
        "source_counts": [],
        "source_separations_m": [],
        "phase_transform_exponents": [0.7],
        "pairwise_combinators": ["product"],
    }
    return argparse.Namespace(**{**defaults, **overrides})


def build_cell(**overrides: object) -> SweepCell:
    defaults = {
        "receiver_count": 4,
        "signal_to_noise_ratio_db": 20.0,
        "source_count": 2,
        "source_separation_m": None,
        "phase_transform_exponent": 0.7,
        "pairwise_combinator": "product",
    }
    return SweepCell(**{**defaults, **overrides})  # type: ignore[arg-type]


def build_outcome(cell: SweepCell, distance_m: float) -> SweepOutcome:
    return SweepOutcome(
        cell=cell,
        estimated_source_count=cell.source_count,
        optimal_subpattern_assignment_m=distance_m,
        median_ellipse_semi_major_m=distance_m,
        record={},
    )


@pytest.fixture(scope="module")
def configuration() -> SimulationConfiguration:
    return load_simulation_configuration(CONFIGURATION_DIRECTORY)


def test_the_grid_is_the_product_of_every_swept_axis() -> None:
    grid = build_sweep_grid(
        build_arguments(
            receiver_counts=[3, 4, 6],
            source_counts=[1, 2],
            pairwise_combinators=["product", "sum"],
        ),
        scene_source_count=2,
    )
    assert len(grid) == 3 * 2 * 2
    assert len({cell.receiver_count for cell in grid}) == 3
    assert len({cell.source_count for cell in grid}) == 2


def test_an_unswept_source_count_falls_back_to_the_whole_scene() -> None:
    grid = build_sweep_grid(build_arguments(), scene_source_count=3)
    assert [cell.source_count for cell in grid] == [3]


def test_an_unswept_separation_leaves_the_scene_positions_alone() -> None:
    grid = build_sweep_grid(build_arguments(), scene_source_count=2)
    assert grid[0].source_separation_m is None


def test_two_sources_are_placed_symmetrically_about_the_domain_centre() -> None:
    sources = (
        SourceConfiguration(position_xy_m=np.array([1.0, 2.0]), amplitude_scale=1.0),
        SourceConfiguration(position_xy_m=np.array([3.0, 4.0]), amplitude_scale=0.5),
    )
    moved = place_two_sources_at_separation(
        sources, DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, 30.0
    )
    assert moved[0].position_xy_m == pytest.approx([35.0, 50.0])
    assert moved[1].position_xy_m == pytest.approx([65.0, 50.0])
    assert float(
        np.linalg.norm(moved[1].position_xy_m - moved[0].position_xy_m)
    ) == pytest.approx(30.0)


def test_the_amplitude_scales_survive_a_separation_sweep() -> None:
    sources = (
        SourceConfiguration(position_xy_m=np.array([1.0, 2.0]), amplitude_scale=1.0),
        SourceConfiguration(position_xy_m=np.array([3.0, 4.0]), amplitude_scale=0.5),
    )
    moved = place_two_sources_at_separation(
        sources, DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, 10.0
    )
    assert [source.amplitude_scale for source in moved] == [1.0, 0.5]


def test_a_separation_sweep_needs_exactly_two_sources() -> None:
    sources = (
        SourceConfiguration(position_xy_m=np.array([1.0, 2.0]), amplitude_scale=1.0),
    )
    with pytest.raises(ValueError, match="two sources only"):
        place_two_sources_at_separation(sources, DOMAIN_EXTENT_M, DOMAIN_EXTENT_M, 10.0)


def test_a_cell_takes_the_first_sources_of_the_scene(
    configuration: SimulationConfiguration,
) -> None:
    sources = select_scene_sources(configuration, build_cell(source_count=1))
    assert len(sources) == 1
    assert sources[0] is configuration.geometry.concurrent_sources[0]


def test_a_cell_cannot_ask_for_more_sources_than_the_scene_declares(
    configuration: SimulationConfiguration,
) -> None:
    with pytest.raises(ValueError, match="fewer than the"):
        select_scene_sources(configuration, build_cell(source_count=99))


def test_a_series_averages_only_the_cells_that_match_it() -> None:
    outcomes = [
        build_outcome(build_cell(receiver_count=3, source_count=1), 4.0),
        build_outcome(build_cell(receiver_count=3, source_count=1), 6.0),
        build_outcome(build_cell(receiver_count=3, source_count=2), 1.0),
        build_outcome(build_cell(receiver_count=6, source_count=1), 2.0),
    ]
    series = summarise_over(
        outcomes,
        np.array([3.0, 6.0]),
        "receiver_count",
        "source_count",
        [1, 2],
        "optimal_subpattern_assignment_m",
    )
    assert series.shape == (2, 2)
    assert series[0, 0] == pytest.approx(5.0)
    assert series[0, 1] == pytest.approx(2.0)
    assert series[1, 0] == pytest.approx(1.0)


def test_a_series_reports_not_a_number_where_no_cell_ran() -> None:
    outcomes = [build_outcome(build_cell(receiver_count=3, source_count=1), 4.0)]
    series = summarise_over(
        outcomes,
        np.array([3.0, 8.0]),
        "receiver_count",
        "source_count",
        [1],
        "optimal_subpattern_assignment_m",
    )
    assert series[0, 0] == pytest.approx(4.0)
    assert np.isnan(series[0, 1])


def test_a_cell_slug_distinguishes_every_swept_axis() -> None:
    first = describe_cell(build_cell(receiver_count=20, source_count=15))
    second = describe_cell(build_cell(receiver_count=20, source_count=10))
    third = describe_cell(build_cell(receiver_count=25, source_count=15))
    assert first != second
    assert first != third
    assert "n20" in first
    assert "k15" in first


def test_a_cell_slug_is_filename_safe() -> None:
    slug = describe_cell(
        build_cell(source_separation_m=2.5, pairwise_combinator="harmonic_mean")
    )
    assert set(slug) <= set(string.ascii_lowercase + string.digits + "_.")
    assert "sep2.5m" in slug


def test_an_unswept_separation_is_named_for_the_scene() -> None:
    assert "sepscene" in describe_cell(build_cell(source_separation_m=None))
