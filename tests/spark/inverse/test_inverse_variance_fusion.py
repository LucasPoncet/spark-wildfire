import numpy as np
import pytest

from spark.inverse.inverse_variance_fusion import fuse_inverse_variance


def test_fusion_favours_the_precise_estimate() -> None:
    fused = fuse_inverse_variance(np.array([1.0, 2.0]), np.array([1e-4, 1.0]))
    assert fused.value == pytest.approx(1.0, abs=1e-3)


def test_fused_variance_is_below_every_input_variance() -> None:
    variances = np.array([0.25, 0.5, 1.0])
    fused = fuse_inverse_variance(np.array([1.0, 1.0, 1.0]), variances)
    assert fused.variance < variances.min()


def test_equal_variances_give_the_plain_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    fused = fuse_inverse_variance(values, np.full(values.size, 0.5))
    assert fused.value == pytest.approx(values.mean())


def test_residuals_are_each_value_minus_the_fused_value() -> None:
    values = np.array([1.0, 2.0, 3.0])
    fused = fuse_inverse_variance(values, np.full(values.size, 0.5))
    assert np.allclose(fused.residuals, values - fused.value)


def test_consistent_inputs_give_unit_reduced_chi_square() -> None:
    variances = np.full(200, 0.25)
    values = np.random.default_rng(0).normal(loc=3.0, scale=0.5, size=200)
    fused = fuse_inverse_variance(values, variances)
    assert fused.reduced_chi_square == pytest.approx(1.0, abs=0.25)


def test_disagreeing_inputs_give_a_large_reduced_chi_square() -> None:
    fused = fuse_inverse_variance(np.array([0.0, 10.0]), np.array([0.01, 0.01]))
    assert fused.reduced_chi_square > 100.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        fuse_inverse_variance(np.array([1.0, 2.0]), np.array([1.0]))


def test_a_single_estimate_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two estimates"):
        fuse_inverse_variance(np.array([1.0]), np.array([1.0]))


def test_non_positive_variance_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fuse_inverse_variance(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
