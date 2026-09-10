import numpy as np

from src.spark.acoustic.exponential_attenuation_channel import (
    compute_atmospheric_absorption_gain,
    compute_geometric_spreading_gain,
)
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.utils.array_types import Float64Array


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


def render_multi_source_receiver_signals(
    source_signals: list[Float64Array],
    source_positions_xyz_m: Float64Array,
    source_amplitude_scales: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    sample_rate_hz: int,
    conditions: AtmosphericConditions,
    reference_distance_m: float,
) -> Float64Array:
    """Renders several concurrent sources to several receivers, all on one clock.

    Superposition is linear, so each source is propagated over each path by the
    same per-pair operator the single-source render uses and the results are
    summed. The single-source render is therefore the `n_sources = 1` case of
    this one, not a different model.

    Args:
        source_signals: One emitted waveform per source.
        source_positions_xyz_m: Source positions, shape `(n_sources, 3)`.
        source_amplitude_scales: Multiplier per source, shape `(n_sources,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        sample_rate_hz: Sample rate in hertz.
        conditions: Air temperature, relative humidity and pressure.
        reference_distance_m: Distance at which spreading gain is unity, in metres.

    Returns:
        Received waveforms of shape `(n_receivers, n_samples)`, zero-padded to
        the longest path so every channel shares one time origin.

    Raises:
        ValueError: If the source count disagrees between signals, positions and
            scales.
    """
    source_positions = np.atleast_2d(
        np.asarray(source_positions_xyz_m, dtype=np.float64)
    )
    receiver_positions = np.atleast_2d(
        np.asarray(receiver_positions_xyz_m, dtype=np.float64)
    )
    amplitude_scales = np.asarray(source_amplitude_scales, dtype=np.float64)
    if not len(source_signals) == source_positions.shape[0] == amplitude_scales.size:
        raise ValueError(
            "source_signals, source_positions_xyz_m and source_amplitude_scales "
            "must describe the same number of sources"
        )

    rendered: list[list[Float64Array]] = []
    for source_signal, source_position, amplitude_scale in zip(
        source_signals, source_positions, amplitude_scales, strict=True
    ):
        distances_m = np.linalg.norm(receiver_positions - source_position, axis=1)
        rendered.append(
            [
                apply_free_field_propagation(
                    amplitude_scale * np.asarray(source_signal, dtype=np.float64),
                    sample_rate_hz,
                    float(distance_m),
                    conditions,
                    reference_distance_m,
                )
                for distance_m in distances_m
            ]
        )

    common_sample_count = max(
        channel.size for channels in rendered for channel in channels
    )
    receiver_signals = np.zeros(
        (receiver_positions.shape[0], common_sample_count), dtype=np.float64
    )
    for channels in rendered:
        for receiver_index, channel in enumerate(channels):
            receiver_signals[receiver_index, : channel.size] += channel
    return receiver_signals
