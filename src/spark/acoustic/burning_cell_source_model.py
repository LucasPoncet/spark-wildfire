"""Turns a FireState into acoustic point sources.

The only bridge between the fire domain and the acoustic domain. A burning
cell becomes one source at its own cell position, with an amplitude proxied
by the fuel mass loss rate. Two emission models are offered: every burning
cell, or only the cells on the advancing front. Knows nothing about
receivers, channels or propagation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.spark.fire.fire_state import FireState
from src.utils.array_types import BoolArray, Float64Array, Int64Array


@dataclass(frozen=True)
class BurningCellSources:
    """Positions, amplitudes and mesh indices of the cells currently alight.

    Attributes:
        source_positions_xy_m: Float64 array of shape (n_burning, 2).
        source_amplitudes: Float64 array of shape (n_burning,).
        burning_cell_indices: Int64 array of shape (n_burning,), indices into
            the mesh's cell arrays.
    """

    source_positions_xy_m: Float64Array
    source_amplitudes: Float64Array
    burning_cell_indices: Int64Array


def extract_burning_cell_sources(
    fire_state: FireState,
    cell_positions_xyz: Float64Array,
    fuel_load_kg_per_m2: float,
    residence_time_s: float,
) -> BurningCellSources:
    """Collect one acoustic source per burning cell.

    Amplitude is the fuel mass loss rate, identical for every burning cell
    while the fuel is uniform, so only geometry drives received levels.

    Args:
        fire_state: State to read burning cells from.
        cell_positions_xyz: Float64 array of shape (cell_count, 3).
        fuel_load_kg_per_m2: Dry fuel mass per unit ground area.
        residence_time_s: How long a cell stays alight.

    Returns:
        The burning cells as sources. Every array is empty when nothing burns.

    Raises:
        ValueError: If the residence time is not positive.
    """
    if residence_time_s <= 0.0:
        raise ValueError("residence time must be positive")

    burning_cell_indices = np.flatnonzero(fire_state.is_burning).astype(np.int64)
    mass_loss_rate_kg_per_m2_s = fuel_load_kg_per_m2 / residence_time_s
    return BurningCellSources(
        source_positions_xy_m=cell_positions_xyz[burning_cell_indices, :2],
        source_amplitudes=np.full(
            burning_cell_indices.shape, mass_loss_rate_kg_per_m2_s, dtype=np.float64
        ),
        burning_cell_indices=burning_cell_indices,
    )


def generate_source_signal_for_burning_cell(
    amplitude: float,
    duration_s: float,
    sample_rate_hz: int,
    seed: int = 42,
) -> Float64Array:
    """Generate one burning cell's source waveform as peak-normalised noise.

    White noise stands in for the broadband crackle of combustion. Peak
    normalisation makes the returned signal's maximum absolute value equal
    the requested amplitude.

    Args:
        amplitude: Peak amplitude of the returned signal.
        duration_s: Signal length, in seconds.
        sample_rate_hz: Samples per second.
        seed: Seed making the waveform reproducible.

    Returns:
        Float64 array of shape (duration_s * sample_rate_hz,).

    Raises:
        ValueError: If the resulting sample count is not positive.
    """
    sample_count = int(duration_s * sample_rate_hz)
    if sample_count <= 0:
        raise ValueError("duration and sample rate must give at least one sample")

    noise = np.random.default_rng(seed).standard_normal(sample_count)
    return np.asarray(amplitude * noise / np.max(np.abs(noise)), dtype=np.float64)


def extract_fire_front_sources(
    fire_state: FireState,
    cell_positions_xyz: Float64Array,
    neighbor_indices: Int64Array,
    fuel_load_kg_per_m2: float,
    residence_time_s: float,
) -> BurningCellSources:
    """Collect one acoustic source per burning cell on the leading edge.

    A burning cell is on the front when at least one of its valid neighbors
    has never ignited. Cells burning in the interior of the burnt area are
    surrounded by already-ignited cells and are dropped, which keeps the
    source set on the advancing perimeter rather than smeared over the scar.

    Args:
        fire_state: State to read burning and ignited cells from.
        cell_positions_xyz: Float64 array of shape (cell_count, 3).
        neighbor_indices: Int64 array of shape (cell_count, max_neighbors),
            -1 in unused slots.
        fuel_load_kg_per_m2: Dry fuel mass per unit ground area.
        residence_time_s: How long a cell stays alight.

    Returns:
        The front cells as sources. Every array is empty when nothing burns.

    Raises:
        ValueError: If the residence time is not positive.
    """
    if residence_time_s <= 0.0:
        raise ValueError("residence time must be positive")

    is_on_front = compute_fire_front_mask(fire_state, neighbor_indices)
    front_cell_indices = np.flatnonzero(is_on_front).astype(np.int64)
    mass_loss_rate_kg_per_m2_s = fuel_load_kg_per_m2 / residence_time_s
    return BurningCellSources(
        source_positions_xy_m=cell_positions_xyz[front_cell_indices, :2],
        source_amplitudes=np.full(
            front_cell_indices.shape, mass_loss_rate_kg_per_m2_s, dtype=np.float64
        ),
        burning_cell_indices=front_cell_indices,
    )


def compute_fire_front_mask(
    fire_state: FireState,
    neighbor_indices: Int64Array,
) -> BoolArray:
    """Mark the burning cells that still have unburnt ground next to them.

    Padding slots hold -1, which would index the last cell, so the validity
    mask is applied alongside the ignition test rather than before it.

    Args:
        fire_state: State to read burning and ignited cells from.
        neighbor_indices: Int64 array of shape (cell_count, max_neighbors),
            -1 in unused slots.

    Returns:
        Bool array of shape (cell_count,), True on the advancing front.
    """
    has_unignited_neighbor = np.any(
        (~fire_state.has_ignited[neighbor_indices]) & (neighbor_indices != -1), axis=1
    )
    return np.asarray(fire_state.is_burning & has_unignited_neighbor, dtype=np.bool_)


def identify_connected_front_components(
    is_on_front: BoolArray,
    neighbor_indices: Int64Array,
) -> Int64Array:
    """Label each separate fire front, largest first.

    Pure graph traversal over the mesh adjacency restricted to front cells.
    Two fronts that never touch through a chain of front cells get distinct
    labels, which is what lets a run report how many separate fires exist.

    Args:
        is_on_front: Bool array of shape (cell_count,).
        neighbor_indices: Int64 array of shape (cell_count, max_neighbors),
            -1 in unused slots.

    Returns:
        Int64 array of shape (cell_count,), -1 off the front and 0, 1, 2, ...
        on it, with 0 the component holding the most cells.
    """
    component_labels = np.full(is_on_front.shape[0], -1, dtype=np.int64)
    front_cell_indices = np.flatnonzero(is_on_front).astype(np.int64)
    if front_cell_indices.size == 0:
        return component_labels

    local_index_of_cell = np.full(is_on_front.shape[0], -1, dtype=np.int64)
    local_index_of_cell[front_cell_indices] = np.arange(
        front_cell_indices.size, dtype=np.int64
    )

    candidate_neighbors = neighbor_indices[front_cell_indices]
    is_edge = (candidate_neighbors >= 0) & is_on_front[candidate_neighbors]
    edge_source_local = np.repeat(
        np.arange(front_cell_indices.size, dtype=np.int64),
        candidate_neighbors.shape[1],
    )[is_edge.ravel()]
    edge_target_local = local_index_of_cell[
        candidate_neighbors.ravel()[is_edge.ravel()]
    ]

    adjacency = coo_matrix(
        (
            np.ones(edge_source_local.size, dtype=np.int64),
            (edge_source_local, edge_target_local),
        ),
        shape=(front_cell_indices.size, front_cell_indices.size),
    )
    component_count, raw_labels = connected_components(adjacency, directed=False)

    component_sizes = np.bincount(raw_labels, minlength=component_count)
    largest_first = np.argsort(-component_sizes, kind="stable")
    rank_of_raw_label = np.empty(component_count, dtype=np.int64)
    rank_of_raw_label[largest_first] = np.arange(component_count, dtype=np.int64)
    component_labels[front_cell_indices] = rank_of_raw_label[raw_labels]
    return component_labels
