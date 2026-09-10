"""Why range is recoverable at all: absorption against frequency and range.

Takes data and returns a `matplotlib.figure.Figure`. Never calls `savefig`.
No existing plotter owns the channel, and without this figure the claim that
the high bands carry the range information is only asserted.
"""

import numpy as np
from matplotlib.figure import Figure

from src.spark.acoustic.exponential_attenuation_channel import (
    compute_atmospheric_absorption_gain,
    compute_geometric_spreading_gain,
)
from src.spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.utils.array_types import Float64Array

DECIBELS_PER_AMPLITUDE_DECADE: float = 20.0
GAIN_FLOOR: float = 1e-12


def compute_total_channel_gain_db(
    band_center_frequencies_hz: Float64Array,
    ranges_m: Float64Array,
    conditions: AtmosphericConditions,
    reference_distance_m: float,
) -> Float64Array:
    """Spreading and absorption gain per band across a sweep of ranges.

    Args:
        band_center_frequencies_hz: One frequency per band, in hertz.
        ranges_m: Source-receiver distances to evaluate, in metres.
        conditions: Air temperature, relative humidity and pressure.
        reference_distance_m: Distance at which spreading gain is unity.

    Returns:
        Float64 array of shape (n_bands, n_ranges), in decibels.
    """
    gains = np.empty((band_center_frequencies_hz.size, ranges_m.size), dtype=np.float64)
    for range_index, range_m in enumerate(ranges_m):
        spreading_gain = compute_geometric_spreading_gain(
            float(range_m), reference_distance_m
        )
        absorption_gain = compute_atmospheric_absorption_gain(
            band_center_frequencies_hz, float(range_m), conditions
        )
        gains[:, range_index] = spreading_gain * absorption_gain
    return np.asarray(
        DECIBELS_PER_AMPLITUDE_DECADE * np.log10(np.maximum(gains, GAIN_FLOOR)),
        dtype=np.float64,
    )


def plot_atmospheric_absorption(
    frequencies_hz: Float64Array,
    conditions: list[AtmosphericConditions],
    condition_labels: list[str],
) -> Figure:
    """ISO 9613-1 absorption against frequency, one curve per air state.

    Args:
        frequencies_hz: Frequencies to evaluate, in hertz.
        conditions: One air state per curve.
        condition_labels: One label per curve.

    Returns:
        A matplotlib Figure. The caller saves or displays it.

    Raises:
        ValueError: If the conditions and labels do not have the same length.
    """
    if len(conditions) != len(condition_labels):
        raise ValueError("every air state needs exactly one label")

    figure = Figure(figsize=(3.5, 2.8))
    axes = figure.add_subplot(111)
    for air_state, label in zip(conditions, condition_labels, strict=True):
        axes.plot(
            frequencies_hz,
            compute_absorption_coefficients_db_per_m(frequencies_hz, air_state),
            label=label,
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("frequency (Hz)")
    axes.set_ylabel(r"absorption $\alpha$ (dB/m)")
    axes.set_title("ISO 9613-1 atmospheric absorption")
    axes.legend(fontsize="x-small")
    axes.grid(True, which="both", linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_channel_gain_against_range(
    band_center_frequencies_hz: Float64Array,
    ranges_m: Float64Array,
    conditions: AtmosphericConditions,
    reference_distance_m: float,
) -> Figure:
    """Channel gain against range, one curve per octave band.

    Args:
        band_center_frequencies_hz: One frequency per band, in hertz.
        ranges_m: Source-receiver distances to evaluate, in metres.
        conditions: Air temperature, relative humidity and pressure.
        reference_distance_m: Distance at which spreading gain is unity.

    Returns:
        A matplotlib Figure. The caller saves or displays it.
    """
    gains_db = compute_total_channel_gain_db(
        band_center_frequencies_hz, ranges_m, conditions, reference_distance_m
    )
    figure = Figure(figsize=(3.5, 2.8))
    axes = figure.add_subplot(111)
    for band_index, frequency_hz in enumerate(band_center_frequencies_hz):
        axes.plot(ranges_m, gains_db[band_index], label=f"{frequency_hz:.0f} Hz")
    axes.set_xlabel("source-receiver range (m)")
    axes.set_ylabel("channel gain (dB)")
    axes.set_title("Spreading and absorption per band")
    axes.legend(fontsize="xx-small", ncol=2)
    axes.grid(True, linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_channel_panels(
    frequencies_hz: Float64Array,
    band_center_frequencies_hz: Float64Array,
    ranges_m: Float64Array,
    conditions: AtmosphericConditions,
    reference_distance_m: float,
) -> Figure:
    """Absorption and per-band range dependence in one two-panel figure.

    The right panel is the one that carries the argument: the spread between
    bands grows with range, and that spread is what an estimator inverts.

    Args:
        frequencies_hz: Frequencies for the absorption panel, in hertz.
        band_center_frequencies_hz: One frequency per band, in hertz.
        ranges_m: Source-receiver distances to evaluate, in metres.
        conditions: Air temperature, relative humidity and pressure.
        reference_distance_m: Distance at which spreading gain is unity.

    Returns:
        A matplotlib Figure with two axes.
    """
    figure = Figure(figsize=(7.0, 2.8))
    absorption_axes = figure.add_subplot(121)
    absorption_axes.plot(
        frequencies_hz,
        compute_absorption_coefficients_db_per_m(frequencies_hz, conditions),
        color="#2f5d8a",
    )
    absorption_axes.set_xscale("log")
    absorption_axes.set_yscale("log")
    absorption_axes.set_xlabel("frequency (Hz)")
    absorption_axes.set_ylabel(r"absorption $\alpha$ (dB/m)")
    absorption_axes.set_title(
        f"ISO 9613-1 at {conditions.air_temperature_celsius:.0f} °C, "
        f"{conditions.relative_humidity_percent:.0f} % RH"
    )
    absorption_axes.grid(True, which="both", linewidth=0.3, alpha=0.5)

    gains_db = compute_total_channel_gain_db(
        band_center_frequencies_hz, ranges_m, conditions, reference_distance_m
    )
    gain_axes = figure.add_subplot(122)
    for band_index, frequency_hz in enumerate(band_center_frequencies_hz):
        gain_axes.plot(ranges_m, gains_db[band_index], label=f"{frequency_hz:.0f} Hz")
    gain_axes.set_xlabel("source-receiver range (m)")
    gain_axes.set_ylabel("channel gain (dB)")
    gain_axes.set_title("Spreading and absorption per band")
    gain_axes.legend(fontsize="xx-small", ncol=2)
    gain_axes.grid(True, linewidth=0.3, alpha=0.5)

    figure.tight_layout()
    return figure
