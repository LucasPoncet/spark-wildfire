import numpy as np

from src.utils.array_types import Float64Array

POWER_FLOOR: float = 1e-20


def make_analysis_window(window_sample_count: int) -> Float64Array:
    return np.asarray(np.hanning(window_sample_count), dtype=np.float64)


def compute_band_level_db(segment: Float64Array, window: Float64Array) -> float:
    weighted_power = np.mean((segment * window) ** 2) / np.mean(window**2)
    return float(10.0 * np.log10(weighted_power + POWER_FLOOR))


def compute_window_start_indices(
    first_index: int,
    last_index: int,
    window_sample_count: int,
    hop_sample_count: int,
) -> list[int]:
    if hop_sample_count <= 0:
        raise ValueError("hop_sample_count must be positive")
    stop = last_index - window_sample_count + 1
    if stop <= first_index:
        return []
    return list(range(first_index, stop, hop_sample_count))
