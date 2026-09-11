import numpy as np
import pytest

from src.spark.inverse.map_normalization import (
    normalize_map_to_unit_mass,
    prepare_map_for_moments,
    subtract_map_background,
)

BACKGROUND_PERCENTILE: float = 60.0


def build_map() -> np.ndarray:
    generator = np.random.default_rng(0)
    return np.abs(generator.standard_normal(4096)) + 0.25


def test_normalisation_gives_unit_mass() -> None:
    assert float(np.sum(normalize_map_to_unit_mass(build_map()))) == pytest.approx(1.0)


def test_normalisation_leaves_an_empty_map_empty() -> None:
    assert np.all(normalize_map_to_unit_mass(np.zeros(64)) == 0.0)


def test_normalisation_does_not_move_the_peak() -> None:
    values = build_map()
    assert int(np.argmax(normalize_map_to_unit_mass(values))) == int(np.argmax(values))


def test_background_subtraction_is_idempotent() -> None:
    once = subtract_map_background(build_map(), BACKGROUND_PERCENTILE)
    twice = subtract_map_background(once, BACKGROUND_PERCENTILE)
    assert twice == pytest.approx(once)


def test_background_subtraction_zeroes_that_share_of_the_frame() -> None:
    subtracted = subtract_map_background(build_map(), BACKGROUND_PERCENTILE)
    zero_fraction = float(np.mean(subtracted == 0.0))
    assert zero_fraction == pytest.approx(BACKGROUND_PERCENTILE / 100.0, abs=0.01)


def test_background_subtraction_never_returns_a_negative_cell() -> None:
    assert np.all(subtract_map_background(build_map(), 95.0) >= 0.0)


def test_a_percentile_outside_the_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        subtract_map_background(build_map(), 120.0)


def test_preparation_is_subtraction_followed_by_normalisation() -> None:
    values = build_map()
    assert prepare_map_for_moments(values, BACKGROUND_PERCENTILE) == pytest.approx(
        normalize_map_to_unit_mass(
            subtract_map_background(values, BACKGROUND_PERCENTILE)
        )
    )


def test_preparation_is_invariant_to_the_scale_of_the_input() -> None:
    values = build_map()
    assert prepare_map_for_moments(values, BACKGROUND_PERCENTILE) == pytest.approx(
        prepare_map_for_moments(1000.0 * values, BACKGROUND_PERCENTILE)
    )
