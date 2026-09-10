from dataclasses import dataclass

import numpy as np

from src.utils.array_types import Float64Array

POWER_FLOOR: float = 1e-20


@dataclass(frozen=True)
class MatchedFilterBandLevels:
    """Per-receiver, per-band level of one source recovered from a mixture.

    Attributes:
        band_center_frequencies_hz: Band centres, shape `(n_bands,)`.
        levels_db: Level of this source at each receiver and band, shape
            `(n_receivers, n_bands)`, in decibels relative to an arbitrary but
            consistent reference.
        window_variance_db2: Variance of that level across analysis windows,
            same shape.
        window_count: Number of analysis windows used.
    """

    band_center_frequencies_hz: Float64Array
    levels_db: Float64Array
    window_variance_db2: Float64Array
    window_count: int


def compute_band_bin_masks(
    frequencies_hz: Float64Array, band_edges_hz: Float64Array
) -> list[Float64Array]:
    """Marks which transform bins fall inside each band.

    Args:
        frequencies_hz: Bin centre frequencies, shape `(n_bins,)`.
        band_edges_hz: Lower and upper edge per band, shape `(n_bands, 2)`.

    Returns:
        One boolean mask over bins per band.

    Raises:
        ValueError: If the edge array is not `(n_bands, 2)`.
    """
    edges_hz = np.asarray(band_edges_hz, dtype=np.float64)
    if edges_hz.ndim != 2 or edges_hz.shape[1] != 2:
        raise ValueError("band_edges_hz must have shape (n_bands, 2)")
    bins_hz = np.asarray(frequencies_hz, dtype=np.float64)
    return [
        np.asarray((bins_hz >= lower_hz) & (bins_hz <= upper_hz), dtype=np.bool_)
        for lower_hz, upper_hz in edges_hz
    ]


def compute_matched_filter_band_levels_db(
    reference_signal: Float64Array,
    receiver_signals: Float64Array,
    band_center_frequencies_hz: Float64Array,
    band_edges_hz: Float64Array,
    sample_rate_hz: int,
    window_sample_count: int,
    overlap_fraction: float,
) -> MatchedFilterBandLevels:
    """Recovers one source's band levels at every receiver from the mixture.

    Projects each receiver channel onto the reference within each band: the
    squared magnitude of the band-restricted cross-spectrum, normalised by the
    reference auto-spectrum, is the power the reference explains in that
    channel. Other sources are rejected to the extent that they are uncorrelated
    with the reference, which is what makes this usable in place of waveform
    subtraction.

    Both the reference and the receiver channels must already sit on a common
    clock: a residual delay puts a phase ramp across each band and cancels part
    of the projection. Aligning to the nearest sample is enough.

    Args:
        reference_signal: Beamformed estimate of one source, shape `(n_samples,)`.
        receiver_signals: Aligned receiver channels, shape
            `(n_receivers, n_samples)`.
        band_center_frequencies_hz: Band centres, shape `(n_bands,)`.
        band_edges_hz: Lower and upper edge per band, shape `(n_bands, 2)`.
        sample_rate_hz: Sample rate in hertz.
        window_sample_count: Length of one analysis window, in samples.
        overlap_fraction: Fraction of a window shared with the previous one.

    Returns:
        The per-receiver, per-band levels with their spread across windows.

    Raises:
        ValueError: If `overlap_fraction` is outside `[0, 1)`, the channels are
            shorter than one window, or the band arrays disagree in length.
    """
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must lie in [0, 1)")
    reference = np.asarray(reference_signal, dtype=np.float64)
    channels = np.atleast_2d(np.asarray(receiver_signals, dtype=np.float64))
    centres_hz = np.asarray(band_center_frequencies_hz, dtype=np.float64)
    edges_hz = np.asarray(band_edges_hz, dtype=np.float64)
    if centres_hz.size != edges_hz.shape[0]:
        raise ValueError("one pair of band edges is required per band centre")

    common_sample_count = min(reference.size, channels.shape[1])
    if common_sample_count < window_sample_count:
        raise ValueError("the signals are shorter than one analysis window")
    reference = reference[:common_sample_count]
    channels = channels[:, :common_sample_count]

    window = np.hanning(window_sample_count)
    hop_sample_count = max(round(window_sample_count * (1.0 - overlap_fraction)), 1)
    starts = list(
        range(0, common_sample_count - window_sample_count + 1, hop_sample_count)
    )
    frequencies_hz = np.fft.rfftfreq(window_sample_count, d=1.0 / sample_rate_hz)
    band_masks = compute_band_bin_masks(frequencies_hz, edges_hz)

    window_levels_db = np.empty(
        (len(starts), channels.shape[0], centres_hz.size), dtype=np.float64
    )
    for window_index, start in enumerate(starts):
        reference_spectrum = np.fft.rfft(
            reference[start : start + window_sample_count] * window
        )
        channel_spectra = np.fft.rfft(
            channels[:, start : start + window_sample_count] * window, axis=1
        )
        cross_spectra = np.conj(reference_spectrum)[None, :] * channel_spectra
        for band_index, band_mask in enumerate(band_masks):
            reference_power = float(np.sum(np.abs(reference_spectrum[band_mask]) ** 2))
            explained_power = np.abs(
                np.sum(cross_spectra[:, band_mask], axis=1)
            ) ** 2 / (reference_power + POWER_FLOOR)
            window_levels_db[window_index, :, band_index] = 10.0 * np.log10(
                explained_power + POWER_FLOOR
            )

    return MatchedFilterBandLevels(
        band_center_frequencies_hz=centres_hz,
        levels_db=np.asarray(window_levels_db.mean(axis=0), dtype=np.float64),
        window_variance_db2=np.asarray(
            window_levels_db.var(axis=0, ddof=1)
            if len(starts) > 1
            else np.zeros_like(window_levels_db[0]),
            dtype=np.float64,
        ),
        window_count=len(starts),
    )
