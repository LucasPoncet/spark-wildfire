import numpy as np

from spark.acoustic.exponential_attenuation_channel import (
    compute_atmospheric_absorption_gain,
    compute_geometric_spreading_gain,
)
from spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from utils.array_types import Float64Array


def apply_free_field_propagation(
    source_signal: Float64Array,
    sample_rate_hz: int,
    source_receiver_distance_m: float,
    conditions: AtmosphericConditions,
    reference_distance_m: float,
) -> Float64Array:
    """Propagates one waveform over one free-field path.

    Applies propagation delay, `1/r` spreading and frequency-dependent ISO 9613-1
    absorption in a single real FFT. The signal is zero-padded by the delay first,
    so the tail never wraps into the head.

    Args:
        source_signal: Emitted waveform.
        sample_rate_hz: Sample rate in hertz.
        source_receiver_distance_m: Path length in metres.
        conditions: Air temperature, relative humidity and pressure.
        reference_distance_m: Distance at which spreading gain is unity, in metres.

    Returns:
        The received waveform, longer than the input by the delay.
    """
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
    """Renders one waveform per receiver, all on a common clock.

    Args:
        source_signal: Emitted waveform.
        sample_rate_hz: Sample rate in hertz.
        source_position_xy_m: Source position `(x, y)` in metres.
        receiver_positions_xy_m: Receiver positions, shape `(n_receivers, 2)`.
        conditions: Air temperature, relative humidity and pressure.
        reference_distance_m: Distance at which spreading gain is unity, in metres.

    Returns:
        Received waveforms of shape `(n_receivers, n_samples)`, zero-padded to the
        length of the farthest receiver so that every channel shares one time origin.
    """
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
