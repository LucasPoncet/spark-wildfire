import numpy as np
from scipy.signal import butter, sosfiltfilt

from utils.array_types import Float64Array

OCTAVE_HALF_WIDTH_FACTOR: float = np.sqrt(2.0)


def compute_octave_band_edges_hz(
    center_frequency_hz: float,
    sample_rate_hz: int,
    maximum_edge_fraction_of_nyquist: float,
) -> tuple[float, float]:
    """Computes the lower and upper edge of one octave band.

    Args:
        center_frequency_hz: ISO band centre frequency in hertz.
        sample_rate_hz: Sample rate in hertz.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may reach.

    Returns:
        `(lower_edge_hz, upper_edge_hz)`. The upper edge is clipped below Nyquist.

    Raises:
        ValueError: If the band does not fit below the Nyquist frequency.
    """
    nyquist_hz = 0.5 * sample_rate_hz
    lower_edge_hz = center_frequency_hz / OCTAVE_HALF_WIDTH_FACTOR
    upper_edge_hz = min(
        center_frequency_hz * OCTAVE_HALF_WIDTH_FACTOR,
        maximum_edge_fraction_of_nyquist * nyquist_hz,
    )
    if lower_edge_hz >= upper_edge_hz:
        raise ValueError(
            f"octave band centred at {center_frequency_hz} Hz does not fit below the "
            f"Nyquist frequency of {nyquist_hz} Hz"
        )
    return float(lower_edge_hz), float(upper_edge_hz)


def design_octave_bandpass(
    center_frequency_hz: float,
    sample_rate_hz: int,
    filter_order: int,
    maximum_edge_fraction_of_nyquist: float,
) -> Float64Array:
    """Designs a Butterworth octave band-pass in second-order-section form.

    Second-order sections rather than transfer-function coefficients because the
    lowest band is very narrow relative to the sample rate, where the `b, a` form
    loses numerical accuracy.

    Args:
        center_frequency_hz: ISO band centre frequency in hertz.
        sample_rate_hz: Sample rate in hertz.
        filter_order: Butterworth order per band edge.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may reach.

    Returns:
        Second-order sections, shape `(n_sections, 6)`.
    """
    lower_edge_hz, upper_edge_hz = compute_octave_band_edges_hz(
        center_frequency_hz, sample_rate_hz, maximum_edge_fraction_of_nyquist
    )
    nyquist_hz = 0.5 * sample_rate_hz
    sections = butter(
        filter_order,
        [lower_edge_hz / nyquist_hz, upper_edge_hz / nyquist_hz],
        btype="band",
        output="sos",
    )
    return np.asarray(sections, dtype=np.float64)


def apply_octave_bandpass(
    signal: Float64Array,
    center_frequency_hz: float,
    sample_rate_hz: int,
    filter_order: int,
    maximum_edge_fraction_of_nyquist: float,
) -> Float64Array:
    """Band-passes a signal with zero phase, so no delay is introduced.

    Args:
        signal: Samples to filter.
        center_frequency_hz: ISO band centre frequency in hertz.
        sample_rate_hz: Sample rate in hertz.
        filter_order: Butterworth order per band edge.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may reach.

    Returns:
        The band-passed signal, same length as `signal`.
    """
    sections = design_octave_bandpass(
        center_frequency_hz,
        sample_rate_hz,
        filter_order,
        maximum_edge_fraction_of_nyquist,
    )
    return np.asarray(
        sosfiltfilt(sections, np.asarray(signal, dtype=np.float64)), dtype=np.float64
    )
