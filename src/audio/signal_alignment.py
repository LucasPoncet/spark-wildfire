import numpy as np

from src.utils.array_types import Float64Array


def shift_signal_by_samples(signal: Float64Array, shift_samples: int) -> Float64Array:
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
    first_index = max(shift_samples, 0)
    last_index = min(sample_count_1, sample_count_2 + shift_samples)
    return first_index, last_index


def align_channel_pair(
    signal_1: Float64Array,
    signal_2: Float64Array,
    shift_samples: int,
) -> tuple[Float64Array, Float64Array]:
    first = np.asarray(signal_1, dtype=np.float64)
    second = shift_signal_by_samples(signal_2, shift_samples)
    first_index, last_index = compute_valid_index_range(first.size, second.size, shift_samples)
    if last_index <= first_index:
        raise ValueError("the alignment shift leaves no overlapping samples")
    return first[first_index:last_index], second[first_index:last_index]
