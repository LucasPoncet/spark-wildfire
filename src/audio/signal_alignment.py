import numpy as np

from src.utils.array_types import Float64Array


def shift_signal_by_samples(signal: Float64Array, shift_samples: int) -> Float64Array:
    """Shifts a signal in time, filling the vacated region with zeros.

    Args:
        signal: Samples to shift.
        shift_samples: Positive delays the signal, negative advances it.

    Returns:
        A shifted copy of the same length as `signal`.
    """
    source = np.asarray(signal, dtype=np.float64)
    if shift_samples == 0:
        return source.copy()
    shifted = np.zeros_like(source)
    if shift_samples > 0:
        if shift_samples < source.size:
            shifted[shift_samples:] = source[: source.size - shift_samples]
        return shifted
    if -shift_samples < source.size:
        shifted[: source.size + shift_samples] = source[-shift_samples:]
    return shifted


def compute_valid_index_range(
    sample_count_1: int,
    sample_count_2: int,
    shift_samples: int,
) -> tuple[int, int]:
    """Computes the index range where both channels hold real samples after a shift.

    Args:
        sample_count_1: Length of the first channel.
        sample_count_2: Length of the second channel before shifting.
        shift_samples: Shift applied to the second channel.

    Returns:
        `(first_index, last_index)`, half-open, excluding the zero-filled region.
    """
    first_index = max(shift_samples, 0)
    last_index = min(sample_count_1, sample_count_2 + shift_samples)
    return first_index, last_index


def align_channel_pair(
    signal_1: Float64Array,
    signal_2: Float64Array,
    shift_samples: int,
) -> tuple[Float64Array, Float64Array]:
    """Puts two channels on a common clock and crops them to their overlap.

    Args:
        signal_1: First channel, taken as the reference clock.
        signal_2: Second channel, shifted onto that clock.
        shift_samples: Shift applied to `signal_2`, from the estimated delay.

    Returns:
        The two aligned channels, cropped to the region where both are real.

    Raises:
        ValueError: If the shift leaves no overlapping samples.
    """
    first = np.asarray(signal_1, dtype=np.float64)
    second = shift_signal_by_samples(signal_2, shift_samples)
    first_index, last_index = compute_valid_index_range(
        first.size, second.size, shift_samples
    )
    if last_index <= first_index:
        raise ValueError("the alignment shift leaves no overlapping samples")
    return first[first_index:last_index], second[first_index:last_index]
