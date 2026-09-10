import numpy as np
import pytest

from src.audio.excerpt_coherence import (
    compute_maximum_normalized_cross_correlation,
    compute_maximum_off_diagonal_coherence,
    compute_pairwise_excerpt_coherence_matrix,
)

INDEPENDENT_NOISE_CEILING: float = 0.05


def test_identical_excerpts_score_one() -> None:
    samples = np.random.default_rng(0).standard_normal(20000)
    assert compute_maximum_normalized_cross_correlation(
        samples, samples
    ) == pytest.approx(1.0, abs=1e-9)


def test_a_scaled_and_shifted_copy_still_scores_one() -> None:
    samples = np.random.default_rng(1).standard_normal(20000)
    shifted = np.concatenate((np.zeros(500), 0.25 * samples))
    assert compute_maximum_normalized_cross_correlation(
        samples, shifted
    ) == pytest.approx(1.0, abs=1e-4)


def test_independent_noise_scores_near_zero() -> None:
    generator = np.random.default_rng(2)
    coherence = compute_maximum_normalized_cross_correlation(
        generator.standard_normal(200000), generator.standard_normal(200000)
    )
    assert coherence < INDEPENDENT_NOISE_CEILING


def test_the_coherence_matrix_is_symmetric_with_a_unit_diagonal() -> None:
    generator = np.random.default_rng(3)
    excerpts = [generator.standard_normal(20000) for _ in range(4)]
    matrix = compute_pairwise_excerpt_coherence_matrix(excerpts)
    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)


def test_the_worst_off_diagonal_ignores_the_diagonal() -> None:
    generator = np.random.default_rng(4)
    shared = generator.standard_normal(20000)
    excerpts = [shared, shared.copy(), generator.standard_normal(20000)]
    assert compute_maximum_off_diagonal_coherence(
        compute_pairwise_excerpt_coherence_matrix(excerpts)
    ) == pytest.approx(1.0, abs=1e-9)


def test_a_single_excerpt_has_no_off_diagonal_coherence() -> None:
    matrix = compute_pairwise_excerpt_coherence_matrix(
        [np.random.default_rng(5).standard_normal(1000)]
    )
    assert compute_maximum_off_diagonal_coherence(matrix) == 0.0


def test_an_empty_excerpt_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one excerpt"):
        compute_pairwise_excerpt_coherence_matrix([])


def test_the_matrix_agrees_with_the_pairwise_function() -> None:
    generator = np.random.default_rng(6)
    excerpts = [generator.standard_normal(30000) for _ in range(4)]
    matrix = compute_pairwise_excerpt_coherence_matrix(excerpts)
    for first in range(4):
        for second in range(first + 1, 4):
            assert matrix[first, second] == pytest.approx(
                compute_maximum_normalized_cross_correlation(
                    excerpts[first], excerpts[second]
                ),
                abs=1e-12,
            )
