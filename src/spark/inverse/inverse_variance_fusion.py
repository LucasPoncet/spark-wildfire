from dataclasses import dataclass

import numpy as np

from src.utils.array_types import Float64Array


@dataclass(frozen=True)
class FusedEstimate:
    value: float
    variance: float
    reduced_chi_square: float
    residuals: Float64Array


def fuse_inverse_variance(
    values: Float64Array, variances: Float64Array
) -> FusedEstimate:
    sample_values = np.asarray(values, dtype=np.float64)
    sample_variances = np.asarray(variances, dtype=np.float64)
    if sample_values.size != sample_variances.size:
        raise ValueError("values and variances must have the same length")
    if sample_values.size < 2:
        raise ValueError("at least two estimates are required to fuse")
    if np.any(sample_variances <= 0.0):
        raise ValueError("variances must be strictly positive")

    weights = 1.0 / sample_variances
    fused_value = float(np.sum(weights * sample_values) / np.sum(weights))
    fused_variance = float(1.0 / np.sum(weights))
    residuals = sample_values - fused_value
    reduced_chi_square = float(
        np.sum(residuals**2 / sample_variances) / (sample_values.size - 1)
    )
    return FusedEstimate(
        value=fused_value,
        variance=fused_variance,
        reduced_chi_square=reduced_chi_square,
        residuals=residuals,
    )
