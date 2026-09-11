"""Where a steered response power map stops adding, and why.

Every tier of the characterisation plan treats the map as a source density
convolved with one response, which requires the map of two sources to be the
sum of their individual maps. This file pins the two places that assumption
meets the shipped map code, so a residual measured by the audit can be
attributed rather than merely observed.

The audit itself is the measurement and lives in `scripts/`; what is asserted
here is the mechanism, which is fast and does not need a render.
"""

import numpy as np
import pytest

from src.spark.inverse.steered_response_power import (
    combine_pairwise_maps,
    pool_curve_over_intervals,
)
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
)

SAMPLE_RATE_HZ: int = 8000
CELL_COUNT: int = 64


def build_curve(values: np.ndarray) -> GeneralizedCrossCorrelationCurve:
    return GeneralizedCrossCorrelationCurve(
        receiver_pair=(0, 1),
        lags_s=np.arange(values.size, dtype=np.float64) / SAMPLE_RATE_HZ,
        values=values,
        sample_rate_hz=SAMPLE_RATE_HZ,
        window_count=1,
        effective_bandwidth_hz=1000.0,
    )


def build_intervals() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(3)
    lower_index = generator.integers(0, 200, CELL_COUNT)
    width = generator.integers(1, 12, CELL_COUNT)
    return (
        lower_index / SAMPLE_RATE_HZ,
        (lower_index + width) / SAMPLE_RATE_HZ,
    )


@pytest.mark.parametrize("pooling", ["sum", "mean"])
def test_interval_pooling_is_linear_in_the_curve(pooling: str) -> None:
    generator = np.random.default_rng(0)
    first = generator.standard_normal(256)
    second = generator.standard_normal(256)
    lower_s, upper_s = build_intervals()

    combined = pool_curve_over_intervals(
        build_curve(2.0 * first + 3.0 * second), lower_s, upper_s, pooling
    )
    separate = 2.0 * pool_curve_over_intervals(
        build_curve(first), lower_s, upper_s, pooling
    ) + 3.0 * pool_curve_over_intervals(build_curve(second), lower_s, upper_s, pooling)
    assert combined == pytest.approx(separate)


def test_maximum_pooling_is_not_linear_in_the_curve() -> None:
    generator = np.random.default_rng(1)
    first = generator.standard_normal(256)
    second = generator.standard_normal(256)
    lower_s, upper_s = build_intervals()

    combined = pool_curve_over_intervals(
        build_curve(first + second), lower_s, upper_s, "max"
    )
    separate = pool_curve_over_intervals(
        build_curve(first), lower_s, upper_s, "max"
    ) + pool_curve_over_intervals(build_curve(second), lower_s, upper_s, "max")
    assert combined != pytest.approx(separate)


def test_an_unknown_pooling_rule_is_rejected() -> None:
    lower_s, upper_s = build_intervals()
    with pytest.raises(ValueError, match="unknown pooling rule"):
        pool_curve_over_intervals(
            build_curve(np.ones(256)), lower_s, upper_s, "volumetric"
        )


def test_the_sum_combinator_is_additive_once_the_pair_maps_share_a_scale() -> None:
    generator = np.random.default_rng(2)
    first = generator.random((6, CELL_COUNT))
    second = first + generator.random((6, CELL_COUNT))
    scaled_first = (first - first.min(axis=1, keepdims=True)) / np.ptp(
        first, axis=1, keepdims=True
    )
    scaled_second = (second - second.min(axis=1, keepdims=True)) / np.ptp(
        second, axis=1, keepdims=True
    )
    assert combine_pairwise_maps(first, "sum") == pytest.approx(
        np.sum(scaled_first, axis=0)
    )
    assert combine_pairwise_maps(second, "sum") == pytest.approx(
        np.sum(scaled_second, axis=0)
    )


def test_the_per_pair_rescaling_is_what_breaks_additivity_under_sum() -> None:
    """The first of the two mechanisms the audit's residual is made of.

    `combine_pairwise_maps` rescales each pair map to its own minimum and
    maximum before merging. Those two statistics belong to the mixture, so the
    scale a pair map is put on depends on every source present, and the
    combined map of two sources is not the sum of the two combined maps even
    when the pooling underneath is perfectly linear.
    """
    generator = np.random.default_rng(4)
    first = generator.random((6, CELL_COUNT))
    second = generator.random((6, CELL_COUNT))
    joint = combine_pairwise_maps(first + second, "sum")
    separate = combine_pairwise_maps(first, "sum") + combine_pairwise_maps(
        second, "sum"
    )
    assert joint != pytest.approx(separate)


def test_the_product_combinator_is_further_from_additive_than_the_sum() -> None:
    """The second mechanism, and the reason detection and imaging differ.

    The product is a geometric mean, so it is multiplicative in the pair maps
    by construction. It is kept for detection precisely because it demands
    agreement across pairs; that is the same property that disqualifies it from
    imaging.
    """
    generator = np.random.default_rng(5)
    first = generator.random((6, CELL_COUNT)) + 0.1
    second = generator.random((6, CELL_COUNT)) + 0.1

    def residual(combinator: str) -> float:
        joint = combine_pairwise_maps(first + second, combinator)
        separate = combine_pairwise_maps(first, combinator) + combine_pairwise_maps(
            second, combinator
        )
        joint_unit = joint / np.sum(joint)
        separate_unit = separate / np.sum(separate)
        return float(
            np.linalg.norm(joint_unit - separate_unit) / np.linalg.norm(joint_unit)
        )

    assert residual("product") > residual("sum")


def test_an_unknown_combinator_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pairwise combinator"):
        combine_pairwise_maps(np.ones((3, CELL_COUNT)), "additive")
