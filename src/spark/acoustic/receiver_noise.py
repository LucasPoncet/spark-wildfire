import numpy as np

from src.utils.array_types import Float64Array


def add_white_noise_at_snr_db(
    signal: Float64Array,
    signal_to_noise_ratio_db: float,
    random_generator: np.random.Generator,
) -> Float64Array:
    clean = np.asarray(signal, dtype=np.float64)
    signal_power = float(np.mean(clean**2))
    if signal_power <= 0.0:
        return clean
    noise_power = signal_power / (10.0 ** (signal_to_noise_ratio_db / 10.0))
    noise = random_generator.normal(0.0, np.sqrt(noise_power), size=clean.shape)
    return clean + noise


def add_white_noise_to_receiver_signals(
    receiver_signals: Float64Array,
    signal_to_noise_ratio_db: float,
    random_generator: np.random.Generator,
) -> Float64Array:
    return np.stack(
        [
            add_white_noise_at_snr_db(
                channel, signal_to_noise_ratio_db, random_generator
            )
            for channel in np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
        ]
    )
