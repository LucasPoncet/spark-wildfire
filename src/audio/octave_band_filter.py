import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.utils.array_types import Float64Array

OCTAVE_HALF_WIDTH_FACTOR: float = np.sqrt(2.0)


def compute_octave_band_edges_hz(
    center_frequency_hz: float,
    sample_rate_hz: int,
    maximum_edge_fraction_of_nyquist: float,
) -> tuple[float, float]:
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
    lower_edge_hz, upper_edge_hz = compute_octave_band_edges_hz(
        center_frequency_hz, sample_rate_hz, maximum_edge_fraction_of_nyquist
    )
    nyquist_hz = 0.5 * sample_rate_hz
    return butter(
        filter_order,
        [lower_edge_hz / nyquist_hz, upper_edge_hz / nyquist_hz],
        btype="band",
        output="sos",
    )


def apply_octave_bandpass(
    signal: Float64Array,
    center_frequency_hz: float,
    sample_rate_hz: int,
    filter_order: int,
    maximum_edge_fraction_of_nyquist: float,
) -> Float64Array:
    sections = design_octave_bandpass(
        center_frequency_hz, sample_rate_hz, filter_order, maximum_edge_fraction_of_nyquist
    )
    return np.asarray(sosfiltfilt(sections, np.asarray(signal, dtype=np.float64)), dtype=np.float64)
