import numpy as np

from src.spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.utils.array_types import Float64Array


def compute_geometric_spreading_gain(
    source_receiver_distance_m: float,
    reference_distance_m: float,
) -> float:
    return reference_distance_m / max(source_receiver_distance_m, reference_distance_m)


def compute_atmospheric_absorption_gain(
    frequencies_hz: Float64Array,
    source_receiver_distance_m: float,
    conditions: AtmosphericConditions,
) -> Float64Array:
    absorption_db = (
        compute_absorption_coefficients_db_per_m(frequencies_hz, conditions)
        * source_receiver_distance_m
    )
    return np.asarray(10.0 ** (-absorption_db / 20.0), dtype=np.float64)


def apply_free_field_propagation(
    source_signal: Float64Array,
    sample_rate_hz: int,
    source_receiver_distance_m: float,
    conditions: AtmosphericConditions,
    reference_distance_m: float,
) -> Float64Array:
    signal = np.asarray(source_signal, dtype=np.float64)
    speed_of_sound_m_per_s = compute_speed_of_sound_m_per_s(
        conditions.air_temperature_celsius
    )
    propagation_delay_s = source_receiver_distance_m / speed_of_sound_m_per_s

    padding_sample_count = int(np.ceil(propagation_delay_s * sample_rate_hz)) + 1
    padded = np.concatenate([signal, np.zeros(padding_sample_count, dtype=np.float64)])

    frequencies_hz = np.fft.rfftfreq(padded.size, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(padded)
    spreading_gain = compute_geometric_spreading_gain(
        source_receiver_distance_m, reference_distance_m
    )
    absorption_gain = compute_atmospheric_absorption_gain(
        frequencies_hz, source_receiver_distance_m, conditions
    )
    delay_phase = np.exp(-2j * np.pi * frequencies_hz * propagation_delay_s)

    return np.fft.irfft(
        spectrum * spreading_gain * absorption_gain * delay_phase, n=padded.size
    )


def render_receiver_signals(
    source_signal: Float64Array,
    sample_rate_hz: int,
    source_position_xy_m: Float64Array,
    receiver_positions_xy_m: Float64Array,
    conditions: AtmosphericConditions,
    reference_distance_m: float,
) -> Float64Array:
    source_position = np.asarray(source_position_xy_m, dtype=np.float64)
    receiver_positions = np.atleast_2d(
        np.asarray(receiver_positions_xy_m, dtype=np.float64)
    )
    distances_m = np.linalg.norm(receiver_positions - source_position, axis=1)

    channels = [
        apply_free_field_propagation(
            source_signal,
            sample_rate_hz,
            float(distance_m),
            conditions,
            reference_distance_m,
        )
        for distance_m in distances_m
    ]
    common_sample_count = max(channel.size for channel in channels)
    return np.stack(
        [
            np.pad(channel, (0, common_sample_count - channel.size))
            for channel in channels
        ]
    )
