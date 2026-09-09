import numpy as np

from src.utils.array_types import Float64Array


def add_white_noise_at_snr_db(
    signal: Float64Array,
    signal_to_noise_ratio_db: float,
    random_generator: np.random.Generator,
) -> Float64Array:
    """Adds white Gaussian noise at a requested signal-to-noise ratio.

    Args:
        signal: Clean samples.
        signal_to_noise_ratio_db: Requested ratio of signal power to noise power.
        random_generator: Seeded generator, so a run is reproducible.

    Returns:
        The noisy signal. A silent input is returned unchanged.
    """
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
    """Adds independent white noise to every receiver channel.

    The noise is drawn separately per channel, so the channels decorrelate as the
    ratio falls, which is what degrades the delay and level estimates.

    Args:
        receiver_signals: Clean channels, shape `(n_receivers, n_samples)`.
        signal_to_noise_ratio_db: Requested ratio of signal power to noise power.
        random_generator: Seeded generator, so a run is reproducible.

    Returns:
        Noisy channels of the same shape.
    """
    return np.stack(
        [
            add_white_noise_at_snr_db(
                channel, signal_to_noise_ratio_db, random_generator
            )
            for channel in np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
        ]
    )
