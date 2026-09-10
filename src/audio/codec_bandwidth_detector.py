import numpy as np
from scipy.ndimage import median_filter

from src.utils.array_types import Float64Array

POWER_FLOOR: float = 1e-30
LTAS_SMOOTHING_BIN_COUNT: int = 9


def compute_long_term_average_spectrum(
    samples: Float64Array,
    sample_rate_hz: int,
    window_sample_count: int,
    overlap_fraction: float,
) -> tuple[Float64Array, Float64Array]:
    """Averages the power spectrum of every analysis window in a recording.

    Args:
        samples: Mono samples.
        sample_rate_hz: Sample rate in hertz.
        window_sample_count: Length of one analysis window, in samples.
        overlap_fraction: Fraction of a window shared with the previous one.

    Returns:
        `(frequencies_hz, level_db)`, both of shape `(window_sample_count // 2 + 1,)`.
        Levels are relative to an arbitrary but consistent reference.

    Raises:
        ValueError: If `overlap_fraction` is outside `[0, 1)` or the recording is
            shorter than one window.
    """
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must lie in [0, 1)")
    signal = np.asarray(samples, dtype=np.float64)
    if signal.size < window_sample_count:
        raise ValueError("recording is shorter than one analysis window")

    window = np.hanning(window_sample_count)
    hop_sample_count = max(round(window_sample_count * (1.0 - overlap_fraction)), 1)
    starts = range(0, signal.size - window_sample_count + 1, hop_sample_count)
    spectra = np.stack(
        [
            np.abs(np.fft.rfft(signal[start : start + window_sample_count] * window))
            ** 2
            for start in starts
        ]
    )
    mean_power = spectra.mean(axis=0)
    return (
        np.fft.rfftfreq(window_sample_count, d=1.0 / sample_rate_hz),
        np.asarray(10.0 * np.log10(mean_power + POWER_FLOOR), dtype=np.float64),
    )


def detect_codec_cutoff_hz(
    samples: Float64Array,
    sample_rate_hz: int,
    window_sample_count: int,
    overlap_fraction: float,
    floor_drop_db: float,
) -> float:
    """Finds the frequency above which a recording carries no real content.

    A lossy encoder brickwalls the signal and leaves its own noise floor above the
    wall. The cutoff is the highest frequency whose long-term level is still within
    `floor_drop_db` of the spectrum's peak; everything above it is encoder noise
    that a phase transform would otherwise amplify to full weight.

    The search runs on a median-smoothed spectrum so a single loud bin above the
    wall cannot set the answer, but the crossing is then refined on the
    unsmoothed spectrum. A centred median filter holds its in-band value for
    half its width past a step edge, so reading the crossing off the smoothed
    curve alone would report the cutoff several bins too high.

    Args:
        samples: Mono samples.
        sample_rate_hz: Sample rate in hertz.
        window_sample_count: Length of one analysis window, in samples.
        overlap_fraction: Fraction of a window shared with the previous one.
        floor_drop_db: Drop below the spectral peak that counts as the floor.

    Returns:
        Cutoff frequency in hertz, at most the Nyquist frequency.
    """
    frequencies_hz, level_db = compute_long_term_average_spectrum(
        samples, sample_rate_hz, window_sample_count, overlap_fraction
    )
    smoothed_level_db = median_filter(
        level_db, size=LTAS_SMOOTHING_BIN_COUNT, mode="nearest"
    )
    threshold_db = float(smoothed_level_db.max()) - floor_drop_db
    smoothed_above_threshold = np.flatnonzero(smoothed_level_db > threshold_db)
    if smoothed_above_threshold.size == 0:
        return float(0.5 * sample_rate_hz)

    last_smoothed_index = int(smoothed_above_threshold[-1])
    raw_above_threshold = np.flatnonzero(
        level_db[: last_smoothed_index + 1] > threshold_db
    )
    if raw_above_threshold.size == 0:
        return float(frequencies_hz[last_smoothed_index])
    return float(frequencies_hz[raw_above_threshold[-1]])
