import numpy as np
import pytest

from spark.acoustic.exponential_attenuation_channel import (
    AtmosphericAbsorptionChannel,
    ExponentialAttenuationChannel,
    compute_atmospheric_absorption_gain,
    compute_geometric_spreading_gain,
    compute_source_receiver_distances_m,
)
from spark.atmosphere.atmospheric_conditions import AtmosphericConditions

SOURCES_XYZ = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
RECEIVERS_XYZ = np.array([[30.0, 0.0, 0.0], [70.0, 0.0, 0.0]])
ATTENUATION_COEFFICIENT_PER_M = 0.01


def test_distances_have_the_source_by_receiver_shape() -> None:
    distances_m = compute_source_receiver_distances_m(SOURCES_XYZ, RECEIVERS_XYZ)
    assert distances_m.shape == (SOURCES_XYZ.shape[0], RECEIVERS_XYZ.shape[0])


def test_distances_are_euclidean() -> None:
    distances_m = compute_source_receiver_distances_m(SOURCES_XYZ, RECEIVERS_XYZ)
    assert distances_m[0, 0] == pytest.approx(30.0)
    assert distances_m[1, 1] == pytest.approx(60.0)


def test_gain_matrix_has_the_source_by_receiver_shape() -> None:
    channel = ExponentialAttenuationChannel(ATTENUATION_COEFFICIENT_PER_M, 1.0)
    gain = channel.compute_gain_matrix(SOURCES_XYZ, RECEIVERS_XYZ)
    assert gain.shape == (SOURCES_XYZ.shape[0], RECEIVERS_XYZ.shape[0])


def test_gain_falls_with_distance() -> None:
    channel = ExponentialAttenuationChannel(ATTENUATION_COEFFICIENT_PER_M, 1.0)
    gain = channel.compute_gain_matrix(np.array([[0.0, 0.0, 0.0]]), RECEIVERS_XYZ)
    assert gain[0, 0] > gain[0, 1]


def test_gain_halves_per_doubling_without_absorption() -> None:
    channel = ExponentialAttenuationChannel(0.0, 1.0)
    gain = channel.compute_gain_matrix(np.array([[0.0, 0.0, 0.0]]), RECEIVERS_XYZ)
    assert gain[0, 0] / gain[0, 1] == pytest.approx(70.0 / 30.0)


def test_gain_is_symmetric_when_source_and_receiver_swap() -> None:
    channel = ExponentialAttenuationChannel(ATTENUATION_COEFFICIENT_PER_M, 1.0)
    forward = channel.compute_gain_matrix(SOURCES_XYZ, RECEIVERS_XYZ)
    reverse = channel.compute_gain_matrix(RECEIVERS_XYZ, SOURCES_XYZ)
    assert np.allclose(forward, reverse.T)


def test_gain_is_clamped_inside_the_reference_distance() -> None:
    channel = ExponentialAttenuationChannel(0.0, 1.0)
    gain = channel.compute_gain_matrix(
        np.array([[0.0, 0.0, 0.0]]), np.array([[0.1, 0.0, 0.0]])
    )
    assert gain[0, 0] <= 1.0


def test_spreading_gain_halves_per_doubling(reference_distance_m: float) -> None:
    near = compute_geometric_spreading_gain(20.0, reference_distance_m)
    far = compute_geometric_spreading_gain(40.0, reference_distance_m)
    assert near / far == pytest.approx(2.0)


def test_absorption_gain_falls_with_frequency(
    atmosphere: AtmosphericConditions,
) -> None:
    gain = compute_atmospheric_absorption_gain(
        np.array([125.0, 1000.0, 8000.0]), 50.0, atmosphere
    )
    assert np.all(np.diff(gain) < 0.0)


def test_absorption_gain_is_negligible_at_low_frequency(
    atmosphere: AtmosphericConditions,
) -> None:
    gain = compute_atmospheric_absorption_gain(np.array([125.0]), 30.0, atmosphere)
    assert gain[0] == pytest.approx(1.0, abs=5e-3)


def test_frequency_dependent_gain_widens_to_three_dimensions(
    atmosphere: AtmosphericConditions,
) -> None:
    frequencies_hz = np.array([125.0, 1000.0, 8000.0])
    channel = AtmosphericAbsorptionChannel(atmosphere, frequencies_hz, 1.0)
    gain = channel.compute_gain_matrix(SOURCES_XYZ, RECEIVERS_XYZ)
    assert gain.shape == (
        SOURCES_XYZ.shape[0],
        RECEIVERS_XYZ.shape[0],
        frequencies_hz.size,
    )


def test_frequency_dependent_gain_decays_faster_at_high_frequency(
    atmosphere: AtmosphericConditions,
) -> None:
    frequencies_hz = np.array([125.0, 8000.0])
    channel = AtmosphericAbsorptionChannel(atmosphere, frequencies_hz, 1.0)
    gain = channel.compute_gain_matrix(np.array([[0.0, 0.0, 0.0]]), RECEIVERS_XYZ)
    low_band_ratio = gain[0, 0, 0] / gain[0, 1, 0]
    high_band_ratio = gain[0, 0, 1] / gain[0, 1, 1]
    assert high_band_ratio > low_band_ratio
