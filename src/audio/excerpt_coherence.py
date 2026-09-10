import numpy as np

from src.utils.array_types import Float64Array

ENERGY_FLOOR: float = 1e-30


def compute_maximum_normalized_cross_correlation(
    signal_a: Float64Array, signal_b: Float64Array
) -> float:
    """Measures how much waveform two excerpts share, at their best alignment.

    The cross-correlation is normalised by the geometric mean of the two
    energies, so identical excerpts score one and excerpts that share nothing
    score near zero however loud either of them is.

    Args:
        signal_a: First excerpt, mean already free to be non-zero.
        signal_b: Second excerpt.

    Returns:
        Peak normalised correlation over all lags, in `[0, 1]`.
    """
    first = np.asarray(signal_a, dtype=np.float64)
    second = np.asarray(signal_b, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()

    transform_length = 1 << int(np.ceil(np.log2(first.size + second.size)))
    correlation = np.fft.irfft(
        np.fft.rfft(first, transform_length)
        * np.conj(np.fft.rfft(second, transform_length)),
        transform_length,
    )
    energy_product = np.sqrt(
        float(np.sum(first**2)) * float(np.sum(second**2)) + ENERGY_FLOOR
    )
    return float(np.max(np.abs(correlation)) / energy_product)


def compute_pairwise_excerpt_coherence_matrix(
    excerpts: list[Float64Array],
) -> Float64Array:
    """Builds the full matrix of shared waveform content between excerpts.

    Two source excerpts that share content put a peak into every receiver pair's
    cross-correlation at a lag no source occupies, and nothing downstream can
    tell that artefact from a real source. This is the pre-flight guard.

    Args:
        excerpts: One excerpt per source.

    Returns:
        Symmetric matrix of shape `(n_excerpts, n_excerpts)` with ones on the
        diagonal.

    Raises:
        ValueError: If fewer than one excerpt is given.
    """
    if not excerpts:
        raise ValueError("at least one excerpt is required")
    excerpt_count = len(excerpts)
    coherence_matrix = np.eye(excerpt_count, dtype=np.float64)
    for first_index in range(excerpt_count):
        for second_index in range(first_index + 1, excerpt_count):
            coherence = compute_maximum_normalized_cross_correlation(
                excerpts[first_index], excerpts[second_index]
            )
            coherence_matrix[first_index, second_index] = coherence
            coherence_matrix[second_index, first_index] = coherence
    return coherence_matrix


def compute_maximum_off_diagonal_coherence(coherence_matrix: Float64Array) -> float:
    """Reports the worst mutual coherence in a matrix, ignoring the diagonal.

    Args:
        coherence_matrix: Square matrix from
            `compute_pairwise_excerpt_coherence_matrix`.

    Returns:
        The largest off-diagonal entry, or zero for a single excerpt.
    """
    matrix = np.asarray(coherence_matrix, dtype=np.float64)
    if matrix.shape[0] < 2:
        return 0.0
    off_diagonal = ~np.eye(matrix.shape[0], dtype=np.bool_)
    return float(np.max(matrix[off_diagonal]))
