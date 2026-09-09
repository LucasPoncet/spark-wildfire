from dataclasses import dataclass

import numpy as np

from spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from utils.array_types import Float64Array


def compute_source_receiver_distances_m(
    source_positions_xyz: Float64Array,
    receiver_positions_xyz: Float64Array,
) -> Float64Array:
    """Computes the Euclidean distance from every source to every receiver.

    Args:
        source_positions_xyz: Source positions, shape `(n_sources, 3)`.
        receiver_positions_xyz: Receiver positions, shape `(n_receivers, 3)`.

    Returns:
        Distances in metres, shape `(n_sources, n_receivers)`.
    """
    sources = np.atleast_2d(np.asarray(source_positions_xyz, dtype=np.float64))
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz, dtype=np.float64))
    return np.linalg.norm(sources[:, None, :] - receivers[None, :, :], axis=-1)


def compute_geometric_spreading_gain(
    source_receiver_distance_m: float,
    reference_distance_m: float,
) -> float:
    """Computes the `1/r` spherical spreading gain, clamped inside the near field.

    Args:
        source_receiver_distance_m: Source-to-receiver distance in metres.
        reference_distance_m: Distance at which the gain is unity, in metres.

    Returns:
        Amplitude gain, dimensionless.
    """
    return reference_distance_m / max(source_receiver_distance_m, reference_distance_m)


def compute_atmospheric_absorption_gain(
    frequencies_hz: Float64Array,
    source_receiver_distance_m: float,
    conditions: AtmosphericConditions,
) -> Float64Array:
    """Computes the ISO 9613-1 absorption gain over one path, per frequency.

    This is the frequency-dependent widening of the scalar `exp(-alpha * r)` term:
    absorption rises steeply with frequency, so high-frequency crackle decays faster
    than low-frequency roar. That spread across bands is what makes range recoverable.

    Args:
        frequencies_hz: Frequencies at which to evaluate absorption, in hertz.
        source_receiver_distance_m: Path length in metres.
        conditions: Air temperature, relative humidity and pressure.

    Returns:
        Amplitude gain per frequency, same shape as `frequencies_hz`.
    """
    absorption_db = (
        compute_absorption_coefficients_db_per_m(frequencies_hz, conditions)
        * source_receiver_distance_m
    )
    return np.asarray(10.0 ** (-absorption_db / 20.0), dtype=np.float64)


@dataclass(frozen=True)
class ExponentialAttenuationChannel:
    """Free-field channel with `gain(r) = exp(-alpha * r) / r`.

    Attributes:
        attenuation_coefficient_per_m: Scalar absorption coefficient in m^-1.
        reference_distance_m: Distance at which spreading gain is unity, in metres.
    """

    attenuation_coefficient_per_m: float
    reference_distance_m: float

    def compute_gain_matrix(
        self,
        source_positions_xyz: Float64Array,
        receiver_positions_xyz: Float64Array,
    ) -> Float64Array:
        """Computes the source-by-receiver gain matrix.

        Args:
            source_positions_xyz: Source positions, shape `(n_sources, 3)`.
            receiver_positions_xyz: Receiver positions, shape `(n_receivers, 3)`.

        Returns:
            Gain matrix of shape `(n_sources, n_receivers)`.
        """
        distances_m = compute_source_receiver_distances_m(
            source_positions_xyz, receiver_positions_xyz
        )
        spreading_gain = self.reference_distance_m / np.maximum(
            distances_m, self.reference_distance_m
        )
        return spreading_gain * np.exp(
            -self.attenuation_coefficient_per_m * distances_m
        )


@dataclass(frozen=True)
class AtmosphericAbsorptionChannel:
    """Frequency-dependent channel whose gain matrix widens to `(src, rec, freq)`.

    The extension of `ExponentialAttenuationChannel` in which `alpha` becomes a
    function of frequency, taken from ISO 9613-1 at the scenario's air conditions
    rather than from a single fitted constant.

    Attributes:
        conditions: Air temperature, relative humidity and pressure.
        frequencies_hz: Frequencies at which the gain is evaluated, in hertz.
        reference_distance_m: Distance at which spreading gain is unity, in metres.
    """

    conditions: AtmosphericConditions
    frequencies_hz: Float64Array
    reference_distance_m: float

    def compute_gain_matrix(
        self,
        source_positions_xyz: Float64Array,
        receiver_positions_xyz: Float64Array,
    ) -> Float64Array:
        """Computes the source-by-receiver-by-frequency gain matrix.

        Args:
            source_positions_xyz: Source positions, shape `(n_sources, 3)`.
            receiver_positions_xyz: Receiver positions, shape `(n_receivers, 3)`.

        Returns:
            Gain matrix of shape `(n_sources, n_receivers, n_frequencies)`.
        """
        distances_m = compute_source_receiver_distances_m(
            source_positions_xyz, receiver_positions_xyz
        )
        spreading_gain = self.reference_distance_m / np.maximum(
            distances_m, self.reference_distance_m
        )
        absorption_db_per_m = compute_absorption_coefficients_db_per_m(
            self.frequencies_hz, self.conditions
        )
        absorption_gain = 10.0 ** (
            -absorption_db_per_m[None, None, :] * distances_m[:, :, None] / 20.0
        )
        return spreading_gain[:, :, None] * absorption_gain
