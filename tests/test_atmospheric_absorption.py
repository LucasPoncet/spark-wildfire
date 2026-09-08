import numpy as np
import pytest

from src.spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)

OCTAVE_BAND_CENTER_FREQUENCIES_HZ = np.array(
    [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
)

REFERENCE_ABSORPTION_DB_PER_M = {
    (15.0, 70.0): [
        0.000376,
        0.001124,
        0.002358,
        0.004079,
        0.008777,
        0.026608,
        0.094962,
    ],
    (20.0, 70.0): [
        0.000335,
        0.001124,
        0.002791,
        0.004978,
        0.009039,
        0.023086,
        0.077633,
    ],
    (10.0, 80.0): [
        0.000373,
        0.001018,
        0.001963,
        0.003566,
        0.008789,
        0.028966,
        0.104565,
    ],
    (25.0, 50.0): [
        0.000394,
        0.001313,
        0.003223,
        0.005677,
        0.010204,
        0.025863,
        0.086573,
    ],
}


@pytest.mark.parametrize(
    ("temperature_celsius", "relative_humidity_percent"),
    list(REFERENCE_ABSORPTION_DB_PER_M),
)
def test_absorption_matches_published_table(
    temperature_celsius: float, relative_humidity_percent: float
) -> None:
    expected = REFERENCE_ABSORPTION_DB_PER_M[
        (temperature_celsius, relative_humidity_percent)
    ]
    computed = compute_absorption_coefficients_db_per_m(
        OCTAVE_BAND_CENTER_FREQUENCIES_HZ,
        AtmosphericConditions(temperature_celsius, relative_humidity_percent),
    )
    assert np.allclose(computed, expected, rtol=5e-3)


def test_absorption_increases_with_frequency() -> None:
    computed = compute_absorption_coefficients_db_per_m(
        OCTAVE_BAND_CENTER_FREQUENCIES_HZ, AtmosphericConditions(15.0, 70.0)
    )
    assert np.all(np.diff(computed) > 0.0)


def test_absorption_is_negligible_at_low_frequency_over_domain_scale() -> None:
    computed = compute_absorption_coefficients_db_per_m(
        np.array([125.0]), AtmosphericConditions(15.0, 70.0)
    )
    assert computed[0] * 30.0 < 0.05


def test_absorption_is_significant_at_high_frequency_over_domain_scale() -> None:
    computed = compute_absorption_coefficients_db_per_m(
        np.array([8000.0]), AtmosphericConditions(15.0, 70.0)
    )
    assert computed[0] * 30.0 > 2.0


def test_speed_of_sound_at_fifteen_celsius() -> None:
    assert compute_speed_of_sound_m_per_s(15.0) == pytest.approx(340.3, abs=0.5)
