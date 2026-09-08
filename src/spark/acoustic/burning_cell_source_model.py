"""Turns a FireState into acoustic point sources.

The only bridge between the fire domain and the acoustic domain. Every
burning cell becomes one source at its own cell position, with an amplitude
proxied by the fuel mass loss rate. Knows nothing about receivers, channels
or propagation.
"""

from dataclasses import dataclass

import numpy as np

from src.spark.fire.fire_state import FireState
from src.utils.array_types import Float64Array, Int64Array


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
