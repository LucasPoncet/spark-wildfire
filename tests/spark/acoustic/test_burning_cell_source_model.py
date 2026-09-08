import numpy as np
import pytest

from src.spark.acoustic.burning_cell_source_model import (
    extract_burning_cell_sources,
    generate_source_signal_for_burning_cell,
)
from src.spark.fire.fire_state import FireState

CELL_COUNT = 12
FUEL_LOAD_KG_PER_M2 = 0.5
RESIDENCE_TIME_S = 20.0
BURNING_CELLS = (0, 5, 10)


def build_cell_positions_xyz() -> np.ndarray:
    return np.stack(
        (
            np.arange(CELL_COUNT, dtype=np.float64),
            np.arange(CELL_COUNT, dtype=np.float64) * 2.0,
            np.zeros(CELL_COUNT, dtype=np.float64),
        ),
        axis=1,
    )


def build_fire_state(burning_cell_indices: tuple[int, ...]) -> FireState:
    is_burning = np.zeros(CELL_COUNT, dtype=np.bool_)
    is_burning[list(burning_cell_indices)] = True
    return FireState(
        ignition_times_s=np.full(CELL_COUNT, np.inf),
        burnout_times_s=np.full(CELL_COUNT, np.inf),
        is_burning=is_burning,
        has_ignited=is_burning.copy(),
        current_time_s=0.0,
    )


def test_no_burning_cells_returns_empty_sources() -> None:
    sources = extract_burning_cell_sources(
        build_fire_state(()),
        build_cell_positions_xyz(),
        FUEL_LOAD_KG_PER_M2,
        RESIDENCE_TIME_S,
    )
    assert sources.source_positions_xy_m.shape == (0, 2)
    assert sources.source_amplitudes.shape == (0,)
    assert sources.burning_cell_indices.shape == (0,)


def test_three_burning_cells_returns_correct_positions() -> None:
    cell_positions_xyz = build_cell_positions_xyz()
    sources = extract_burning_cell_sources(
        build_fire_state(BURNING_CELLS),
        cell_positions_xyz,
        FUEL_LOAD_KG_PER_M2,
        RESIDENCE_TIME_S,
    )
    np.testing.assert_array_equal(
        sources.source_positions_xy_m, cell_positions_xyz[list(BURNING_CELLS), :2]
    )
    np.testing.assert_array_equal(
        sources.burning_cell_indices, np.array(BURNING_CELLS, dtype=np.int64)
    )
    np.testing.assert_allclose(
        sources.source_amplitudes, FUEL_LOAD_KG_PER_M2 / RESIDENCE_TIME_S
    )


def test_amplitude_is_the_mass_loss_rate() -> None:
    sources = extract_burning_cell_sources(
        build_fire_state(BURNING_CELLS),
        build_cell_positions_xyz(),
        1.0,
        4.0,
    )
    np.testing.assert_allclose(sources.source_amplitudes, 0.25)


def test_non_positive_residence_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="residence time must be positive"):
        extract_burning_cell_sources(
            build_fire_state(BURNING_CELLS), build_cell_positions_xyz(), 0.5, 0.0
        )


def test_generated_signal_has_correct_length_and_amplitude() -> None:
    signal = generate_source_signal_for_burning_cell(
        amplitude=0.025, duration_s=5.0, sample_rate_hz=44100
    )
    assert signal.shape == (220500,)
    assert np.max(np.abs(signal)) <= 0.025
    assert np.max(np.abs(signal)) == pytest.approx(0.025)


def test_generated_signal_is_reproducible_per_seed() -> None:
    first = generate_source_signal_for_burning_cell(1.0, 0.01, 44100, seed=7)
    second = generate_source_signal_for_burning_cell(1.0, 0.01, 44100, seed=7)
    different = generate_source_signal_for_burning_cell(1.0, 0.01, 44100, seed=8)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_empty_signal_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        generate_source_signal_for_burning_cell(1.0, 0.0, 44100)
