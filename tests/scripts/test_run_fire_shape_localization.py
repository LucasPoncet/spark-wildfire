"""The arithmetic the fire-shape runner adds on top of the two pipelines.

Everything here is silent when wrong. A frame captioned `t = 20 s` showing the
fire at 20.6 s looks perfectly reasonable, and a coverage figure that counts the
same estimate twice reads as a better result rather than as a bug.
"""

import numpy as np
import pytest

from scripts.run_fire_shape_localization import (
    advance_fire_to_time_s,
    compute_distances_to_front_m,
    compute_front_coverage_fraction,
    lift_sources_to_height_xyz_m,
)
from src.spark.acoustic.burning_cell_source_model import BurningCellSources
from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fire.cellular_automaton_spread_engine import (
    CellularAutomatonSpreadEngine,
    CellularAutomatonSpreadEngineConfig,
)
from src.spark.fire.fire_state import FireState
from src.spark.terrain.square_grid_mesh import SquareGridMesh

FRONT_XY_M = np.array([[10.0, 10.0], [11.0, 10.0], [12.0, 10.0], [13.0, 10.0]])
SOURCE_HEIGHT_M: float = 0.5
BURN_DURATION_S: float = 20.0


def build_engine_and_state() -> tuple[CellularAutomatonSpreadEngine, FireState]:
    mesh = SquareGridMesh.from_values(10.0, 10.0, 1.0, False)
    engine = CellularAutomatonSpreadEngine(
        CellularAutomatonSpreadEngineConfig(
            random_seed=0, burn_duration_s=BURN_DURATION_S
        )
    )
    state = engine.initialize(
        mesh,
        UniformScalarField(1.0),
        ConstantWindField.from_speed_and_bearing(0.0, 0.0),
    )
    return engine, engine.ignite_cells(state, np.array([0], dtype=np.int64))


def test_the_clock_lands_exactly_on_the_observation_time() -> None:
    engine, state = build_engine_and_state()
    advanced = advance_fire_to_time_s(engine, state, 5.0, 1.83)
    assert advanced.current_time_s == pytest.approx(5.0)


def test_a_step_longer_than_the_window_does_not_overshoot_it() -> None:
    engine, state = build_engine_and_state()
    advanced = advance_fire_to_time_s(engine, state, 5.0, 100.0)
    assert advanced.current_time_s == pytest.approx(5.0)


def test_successive_windows_land_on_their_own_times() -> None:
    engine, state = build_engine_and_state()
    for window_index in range(1, 4):
        state = advance_fire_to_time_s(engine, state, window_index * 5.0, 1.83)
        assert state.current_time_s == pytest.approx(window_index * 5.0)


def test_a_target_already_reached_advances_nothing() -> None:
    engine, state = build_engine_and_state()
    advanced = advance_fire_to_time_s(engine, state, 0.0, 1.83)
    assert advanced.current_time_s == pytest.approx(state.current_time_s)


def test_an_estimate_sitting_on_a_burning_cell_is_zero_from_the_front() -> None:
    distances_m = compute_distances_to_front_m(np.array([[12.0, 10.0]]), FRONT_XY_M)
    assert distances_m == pytest.approx([0.0])


def test_an_estimate_off_the_front_is_measured_to_its_nearest_cell() -> None:
    distances_m = compute_distances_to_front_m(np.array([[13.0, 13.0]]), FRONT_XY_M)
    assert distances_m == pytest.approx([3.0])


def test_a_window_that_located_nothing_has_no_distances_to_report() -> None:
    assert compute_distances_to_front_m(np.empty((0, 2)), FRONT_XY_M).size == 0


def test_coverage_counts_the_front_and_not_the_estimates() -> None:
    two_estimates_on_one_cell = np.array([[10.0, 10.0], [10.1, 10.0]])
    assert compute_front_coverage_fraction(
        two_estimates_on_one_cell, FRONT_XY_M, 1.0
    ) == pytest.approx(0.5)


def test_one_estimate_reaching_every_cell_covers_the_whole_front() -> None:
    assert compute_front_coverage_fraction(
        np.array([[11.5, 10.0]]), FRONT_XY_M, 10.0
    ) == pytest.approx(1.0)


def test_an_estimate_out_of_reach_covers_nothing() -> None:
    assert compute_front_coverage_fraction(
        np.array([[50.0, 50.0]]), FRONT_XY_M, 4.0
    ) == pytest.approx(0.0)


def test_a_window_that_located_nothing_covers_nothing() -> None:
    assert compute_front_coverage_fraction(
        np.empty((0, 2)), FRONT_XY_M, 4.0
    ) == pytest.approx(0.0)


def test_radiating_cells_are_lifted_to_one_common_flame_height() -> None:
    sources = BurningCellSources(
        source_positions_xy_m=FRONT_XY_M,
        source_amplitudes=np.ones(FRONT_XY_M.shape[0]),
        burning_cell_indices=np.arange(FRONT_XY_M.shape[0], dtype=np.int64),
    )
    positions_xyz_m = lift_sources_to_height_xyz_m(sources, SOURCE_HEIGHT_M)
    assert positions_xyz_m.shape == (FRONT_XY_M.shape[0], 3)
    assert positions_xyz_m[:, :2] == pytest.approx(FRONT_XY_M)
    assert positions_xyz_m[:, 2] == pytest.approx(SOURCE_HEIGHT_M)
