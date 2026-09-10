import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.utils.array_types import Float64Array

OCTAVE_HALF_WIDTH_FACTOR: float = np.sqrt(2.0)
FULL_OCTAVE_FRACTION_DENOMINATOR: int = 1
THIRD_OCTAVE_FRACTION_DENOMINATOR: int = 3
ISO_PREFERRED_THIRD_OCTAVE_CENTRE_FREQUENCIES_HZ: tuple[float, ...] = (
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
)
ISO_PREFERRED_REFERENCE_INDEX: int = 16


def compute_band_half_width_factor(fraction_denominator: int) -> float:
    """Computes the factor separating a band centre from either of its edges.

    Args:
        fraction_denominator: Bands per octave. One gives full octaves, three
            gives third octaves.

    Returns:
        The multiplicative factor `2 ** (1 / (2 * fraction_denominator))`.

    Raises:
        ValueError: If the denominator is not positive.
    """
    if fraction_denominator <= 0:
        raise ValueError("fraction_denominator must be positive")
    return float(2.0 ** (1.0 / (2.0 * fraction_denominator)))


def build_fractional_octave_centre_frequencies_hz(
    lowest_centre_hz: float,
    highest_centre_hz: float,
    fraction_denominator: int,
) -> tuple[float, ...]:
    """Lists the ISO preferred band centres inside a frequency range.

    Args:
        lowest_centre_hz: Lowest centre to keep, inclusive.
        highest_centre_hz: Highest centre to keep, inclusive.
        fraction_denominator: Bands per octave, one or three.

    Returns:
        The preferred centres in ascending order.

    Raises:
        ValueError: If the denominator is neither one nor three, or the range
            holds no preferred centre.
    """
    if fraction_denominator not in (
        FULL_OCTAVE_FRACTION_DENOMINATOR,
        THIRD_OCTAVE_FRACTION_DENOMINATOR,
    ):
        raise ValueError(
            "the ISO preferred series is tabulated for one and three bands per "
            f"octave, not {fraction_denominator}"
        )
    stride = THIRD_OCTAVE_FRACTION_DENOMINATOR // fraction_denominator
    centres_hz = tuple(
        centre_hz
        for index, centre_hz in enumerate(
            ISO_PREFERRED_THIRD_OCTAVE_CENTRE_FREQUENCIES_HZ
        )
        if (index - ISO_PREFERRED_REFERENCE_INDEX) % stride == 0
        and lowest_centre_hz <= centre_hz <= highest_centre_hz
    )
    if not centres_hz:
        raise ValueError(
            f"no ISO preferred centre lies between {lowest_centre_hz} Hz and "
            f"{highest_centre_hz} Hz"
        )
    return centres_hz


def compute_fractional_octave_band_edges_hz(
    center_frequency_hz: float,
    fraction_denominator: int,
    sample_rate_hz: int,
    maximum_edge_fraction_of_nyquist: float,
) -> tuple[float, float]:
    """Computes the lower and upper edge of one fractional-octave band.

    Args:
        center_frequency_hz: ISO band centre frequency in hertz.
        fraction_denominator: Bands per octave. One reproduces the octave edges.
        sample_rate_hz: Sample rate in hertz.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may reach.

    Returns:
        `(lower_edge_hz, upper_edge_hz)`. The upper edge is clipped below Nyquist.

    Raises:
        ValueError: If the band does not fit below the Nyquist frequency.
    """
    half_width_factor = compute_band_half_width_factor(fraction_denominator)
    nyquist_hz = 0.5 * sample_rate_hz
    lower_edge_hz = center_frequency_hz / half_width_factor
    upper_edge_hz = min(
        center_frequency_hz * half_width_factor,
        maximum_edge_fraction_of_nyquist * nyquist_hz,
    )
    if lower_edge_hz >= upper_edge_hz:
        raise ValueError(
            f"band centred at {center_frequency_hz} Hz does not fit below the "
            f"Nyquist frequency of {nyquist_hz} Hz"
        )
    return float(lower_edge_hz), float(upper_edge_hz)


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
    """
    return compute_fractional_octave_band_edges_hz(
        center_frequency_hz,
        FULL_OCTAVE_FRACTION_DENOMINATOR,
        sample_rate_hz,
        maximum_edge_fraction_of_nyquist,
    )


def design_fractional_octave_bandpass(
    center_frequency_hz: float,
    fraction_denominator: int,
    sample_rate_hz: int,
    filter_order: int,
    maximum_edge_fraction_of_nyquist: float,
) -> Float64Array:
    """Designs a Butterworth fractional-octave band-pass in second-order-section form.

    Second-order sections rather than transfer-function coefficients because the
    lowest band is very narrow relative to the sample rate, where the `b, a` form
    loses numerical accuracy.

    Args:
        center_frequency_hz: ISO band centre frequency in hertz.
        fraction_denominator: Bands per octave.
        sample_rate_hz: Sample rate in hertz.
        filter_order: Butterworth order per band edge.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may reach.

    Returns:
        Second-order sections, shape `(n_sections, 6)`.
    """
    lower_edge_hz, upper_edge_hz = compute_fractional_octave_band_edges_hz(
        center_frequency_hz,
        fraction_denominator,
        sample_rate_hz,
        maximum_edge_fraction_of_nyquist,
    )
    nyquist_hz = 0.5 * sample_rate_hz
    sections = butter(
        filter_order,
        [lower_edge_hz / nyquist_hz, upper_edge_hz / nyquist_hz],
        btype="band",
        output="sos",
    )
    return np.asarray(sections, dtype=np.float64)


def design_octave_bandpass(
    center_frequency_hz: float,
    sample_rate_hz: int,
    filter_order: int,
    maximum_edge_fraction_of_nyquist: float,
) -> Float64Array:
    """Designs a Butterworth octave band-pass in second-order-section form.

    Args:
        center_frequency_hz: ISO band centre frequency in hertz.
        sample_rate_hz: Sample rate in hertz.
        filter_order: Butterworth order per band edge.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may reach.

    Returns:
        Second-order sections, shape `(n_sections, 6)`.
    """
    return design_fractional_octave_bandpass(
        center_frequency_hz,
        FULL_OCTAVE_FRACTION_DENOMINATOR,
        sample_rate_hz,
        filter_order,
        maximum_edge_fraction_of_nyquist,
    )


def apply_fractional_octave_bandpass(
    signal: Float64Array,
    center_frequency_hz: float,
    fraction_denominator: int,
    sample_rate_hz: int,
    filter_order: int,
    maximum_edge_fraction_of_nyquist: float,
) -> Float64Array:
    """Band-passes a signal with zero phase, so no delay is introduced.

    Args:
        signal: Samples to filter.
        center_frequency_hz: ISO band centre frequency in hertz.
        fraction_denominator: Bands per octave.
        sample_rate_hz: Sample rate in hertz.
        filter_order: Butterworth order per band edge.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may reach.

    Returns:
        The band-passed signal, same length as `signal`.
    """
    sections = design_fractional_octave_bandpass(
        center_frequency_hz,
        fraction_denominator,
        sample_rate_hz,
        filter_order,
        maximum_edge_fraction_of_nyquist,
    )
    return np.asarray(
        sosfiltfilt(sections, np.asarray(signal, dtype=np.float64)), dtype=np.float64
    )


def apply_octave_bandpass(
    signal: Float64Array,
    center_frequency_hz: float,
    sample_rate_hz: int,
    filter_order: int,
    maximum_edge_fraction_of_nyquist: float,
) -> Float64Array:
    """Band-passes a signal with an octave-wide zero-phase filter.

    Args:
        signal: Samples to filter.
        center_frequency_hz: ISO band centre frequency in hertz.
        sample_rate_hz: Sample rate in hertz.
        filter_order: Butterworth order per band edge.
        maximum_edge_fraction_of_nyquist: Fraction of Nyquist the upper edge may reach.

    Returns:
        The band-passed signal, same length as `signal`.
    """
    return apply_fractional_octave_bandpass(
        signal,
        center_frequency_hz,
        FULL_OCTAVE_FRACTION_DENOMINATOR,
        sample_rate_hz,
        filter_order,
        maximum_edge_fraction_of_nyquist,
    )
