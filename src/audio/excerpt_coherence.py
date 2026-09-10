import numpy as np

from src.utils.array_types import ComplexArray, Float64Array

ENERGY_FLOOR: float = 1e-30


def compute_transform_length(sample_counts: tuple[int, ...]) -> int:
    """Sizes a transform so no correlation of these lengths can wrap around.

    Args:
        sample_counts: Lengths that will be correlated against each other.

    Returns:
        A power of two at least as long as the longest linear correlation.
    """
    longest = max(sample_counts)
    return 1 << int(np.ceil(np.log2(2 * longest)))


def prepare_excerpt_spectrum(
    signal: Float64Array, transform_length: int
) -> tuple[ComplexArray, float]:
    """Centres one excerpt and transforms it, keeping its energy.

    Args:
        signal: Excerpt samples.
        transform_length: Transform length from `compute_transform_length`.

    Returns:
        `(spectrum, energy)` of the mean-removed excerpt.
    """
    centred = np.asarray(signal, dtype=np.float64)
    centred = centred - centred.mean()
    return (
        np.asarray(np.fft.rfft(centred, transform_length), dtype=np.complex128),
        float(np.sum(centred**2)),
    )


def compute_coherence_from_spectra(
    spectrum_a: ComplexArray,
    energy_a: float,
    spectrum_b: ComplexArray,
    energy_b: float,
    transform_length: int,
) -> float:
    """Correlates two prepared excerpts and normalises the peak.

    Args:
        spectrum_a: Transform of the first excerpt.
        energy_a: Energy of the first excerpt.
        spectrum_b: Transform of the second excerpt.
        energy_b: Energy of the second excerpt.
        transform_length: Transform length both spectra were taken at.

    Returns:
        Peak normalised correlation over all lags, in `[0, 1]`.
    """
    correlation = np.fft.irfft(spectrum_a * np.conj(spectrum_b), transform_length)
    return float(
        np.max(np.abs(correlation)) / np.sqrt(energy_a * energy_b + ENERGY_FLOOR)
    )


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
    transform_length = compute_transform_length(
        (np.asarray(signal_a).size, np.asarray(signal_b).size)
    )
    spectrum_a, energy_a = prepare_excerpt_spectrum(signal_a, transform_length)
    spectrum_b, energy_b = prepare_excerpt_spectrum(signal_b, transform_length)
    return compute_coherence_from_spectra(
        spectrum_a, energy_a, spectrum_b, energy_b, transform_length
    )


def compute_pairwise_excerpt_coherence_matrix(
    excerpts: list[Float64Array],
) -> Float64Array:
    """Builds the full matrix of shared waveform content between excerpts.

    Two source excerpts that share content put a peak into every receiver pair's
    cross-correlation at a lag no source occupies, and nothing downstream can
    tell that artefact from a real source. This is the pre-flight guard.

    Each row transforms its own excerpt once and reuses that spectrum across the
    row, which is what keeps a sixty-file pool affordable. Every entry is
    identical to calling `compute_maximum_normalized_cross_correlation` on the
    pair, because zero-padding past the linear correlation length cannot change
    where the peak falls or how tall it is.

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
    transform_length = compute_transform_length(
        tuple(np.asarray(excerpt).size for excerpt in excerpts)
    )
    coherence_matrix = np.eye(excerpt_count, dtype=np.float64)
    for first_index in range(excerpt_count):
        spectrum_a, energy_a = prepare_excerpt_spectrum(
            excerpts[first_index], transform_length
        )
        for second_index in range(first_index + 1, excerpt_count):
            spectrum_b, energy_b = prepare_excerpt_spectrum(
                excerpts[second_index], transform_length
            )
            coherence = compute_coherence_from_spectra(
                spectrum_a, energy_a, spectrum_b, energy_b, transform_length
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
