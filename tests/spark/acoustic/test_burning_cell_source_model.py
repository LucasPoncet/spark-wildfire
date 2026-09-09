import numpy as np
import pytest

from src.spark.acoustic.burning_cell_source_model import (
    extract_burning_cell_sources,
    extract_fire_front_sources,
    generate_source_signal_for_burning_cell,
    identify_connected_front_components,
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


GRID_SIDE = 5
LARGE_GRID_SIDE = 10
CARDINAL_OFFSETS_RC = ((0, 1), (0, -1), (1, 0), (-1, 0))


def build_square_neighbor_indices(side: int) -> np.ndarray:
    cell_count = side * side
    neighbor_indices = np.full(
        (cell_count, len(CARDINAL_OFFSETS_RC)), -1, dtype=np.int64
    )
    for cell_index in range(cell_count):
        row, column = divmod(cell_index, side)
        for slot, (row_offset, column_offset) in enumerate(CARDINAL_OFFSETS_RC):
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if 0 <= neighbor_row < side and 0 <= neighbor_column < side:
                neighbor_indices[cell_index, slot] = (
                    neighbor_row * side + neighbor_column
                )
    return neighbor_indices


def build_square_positions_xyz(side: int) -> np.ndarray:
    cell_indices = np.arange(side * side)
    return np.stack(
        (
            (cell_indices % side).astype(np.float64),
            (cell_indices // side).astype(np.float64),
            np.zeros(side * side, dtype=np.float64),
        ),
        axis=1,
    )


def build_square_fire_state(
    side: int, burning_rows_columns: tuple[tuple[int, int], ...]
) -> FireState:
    cell_count = side * side
    is_burning = np.zeros(cell_count, dtype=np.bool_)
    for row, column in burning_rows_columns:
        is_burning[row * side + column] = True
    return FireState(
        ignition_times_s=np.full(cell_count, np.inf),
        burnout_times_s=np.full(cell_count, np.inf),
        is_burning=is_burning,
        has_ignited=is_burning.copy(),
        current_time_s=0.0,
    )


def mask_from_rows_columns(
    side: int, rows_columns: tuple[tuple[int, int], ...]
) -> np.ndarray:
    mask = np.zeros(side * side, dtype=np.bool_)
    for row, column in rows_columns:
        mask[row * side + column] = True
    return mask


def test_front_extraction_excludes_interior_cells() -> None:
    block = tuple((row, column) for row in range(1, 4) for column in range(1, 4))
    sources = extract_fire_front_sources(
        build_square_fire_state(GRID_SIDE, block),
        build_square_positions_xyz(GRID_SIDE),
        build_square_neighbor_indices(GRID_SIDE),
        FUEL_LOAD_KG_PER_M2,
        RESIDENCE_TIME_S,
    )
    interior_cell_index = 2 * GRID_SIDE + 2
    assert sources.burning_cell_indices.size == 8
    assert interior_cell_index not in sources.burning_cell_indices.tolist()
    assert sources.source_positions_xy_m.shape == (8, 2)


def test_connected_components_finds_two_separate_fires() -> None:
    front_cells = ((1, 1), (1, 2), (2, 1), (7, 7), (7, 8), (8, 7))
    is_on_front = mask_from_rows_columns(LARGE_GRID_SIDE, front_cells)
    labels = identify_connected_front_components(
        is_on_front, build_square_neighbor_indices(LARGE_GRID_SIDE)
    )
    assert sorted(set(labels[is_on_front].tolist())) == [0, 1]
    assert np.bincount(labels[is_on_front]).tolist() == [3, 3]
    assert np.all(labels[~is_on_front] == -1)


def test_connected_components_single_fire_returns_one_component() -> None:
    l_shaped_front = ((3, 3), (4, 3), (5, 3), (5, 4), (5, 5))
    is_on_front = mask_from_rows_columns(LARGE_GRID_SIDE, l_shaped_front)
    labels = identify_connected_front_components(
        is_on_front, build_square_neighbor_indices(LARGE_GRID_SIDE)
    )
    assert sorted(set(labels[is_on_front].tolist())) == [0]
    assert np.count_nonzero(labels == 0) == 5


def test_connected_components_orders_the_largest_first() -> None:
    front_cells = ((0, 0), (5, 5), (5, 6), (6, 5), (6, 6))
    is_on_front = mask_from_rows_columns(LARGE_GRID_SIDE, front_cells)
    labels = identify_connected_front_components(
        is_on_front, build_square_neighbor_indices(LARGE_GRID_SIDE)
    )
    assert np.count_nonzero(labels == 0) == 4
    assert np.count_nonzero(labels == 1) == 1


def test_front_extraction_drops_a_fully_enclosed_burning_cell() -> None:
    block = tuple((row, column) for row in range(1, 4) for column in range(1, 4))
    state = build_square_fire_state(GRID_SIDE, block)
    interior_only = FireState(
        ignition_times_s=state.ignition_times_s,
        burnout_times_s=state.burnout_times_s,
        is_burning=mask_from_rows_columns(GRID_SIDE, ((2, 2),)),
        has_ignited=state.has_ignited,
        current_time_s=0.0,
    )
    sources = extract_fire_front_sources(
        interior_only,
        build_square_positions_xyz(GRID_SIDE),
        build_square_neighbor_indices(GRID_SIDE),
        FUEL_LOAD_KG_PER_M2,
        RESIDENCE_TIME_S,
    )
    assert sources.burning_cell_indices.size == 0
