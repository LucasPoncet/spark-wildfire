import numpy as np

from utils.array_types import Float64Array

POWER_FLOOR: float = 1e-20


def make_analysis_window(window_sample_count: int) -> Float64Array:
    """Builds the Hann analysis window used for every band level estimate.

    Args:
        window_sample_count: Window length in samples.

    Returns:
        Hann window of length `window_sample_count`.
    """
    return np.asarray(np.hanning(window_sample_count), dtype=np.float64)


def compute_band_level_db(segment: Float64Array, window: Float64Array) -> float:
    """Computes the windowed RMS level of one band-passed segment.

    The window's own power is divided out, so the level does not depend on the
    window shape. The reference pressure cancels when two levels are subtracted,
    which is the only way this value is ever used.

    Args:
        segment: Band-passed samples, same length as `window`.
        window: Analysis window from `make_analysis_window`.

    Returns:
        Level in decibels relative to an arbitrary but consistent reference.
    """
    weighted_power = np.mean((segment * window) ** 2) / np.mean(window**2)
    return float(10.0 * np.log10(weighted_power + POWER_FLOOR))


def compute_window_start_indices(
    first_index: int,
    last_index: int,
    window_sample_count: int,
    hop_sample_count: int,
) -> list[int]:
    """Lists the start index of every whole window inside a half-open range.

    Args:
        first_index: First usable sample index, inclusive.
        last_index: Last usable sample index, exclusive.
        window_sample_count: Window length in samples.
        hop_sample_count: Distance between consecutive window starts, in samples.

    Returns:
        Start indices, empty when the range cannot hold one whole window.

    Raises:
        ValueError: If `hop_sample_count` is not positive.
    """
    if hop_sample_count <= 0:
        raise ValueError("hop_sample_count must be positive")
    stop = last_index - window_sample_count + 1
    if stop <= first_index:
        return []
    return list(range(first_index, stop, hop_sample_count))
