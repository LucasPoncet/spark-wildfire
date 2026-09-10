import numpy as np

from src.audio.octave_band_filter import compute_band_half_width_factor
from src.utils.array_types import BoolArray, Float64Array


def select_usable_band_centres_hz(
    candidate_centre_frequencies_hz: tuple[float, ...],
    codec_cutoff_hz: float,
    codec_cutoff_margin: float,
    fraction_denominator: int,
) -> tuple[float, ...]:
    """Keeps the bands that lie wholly inside a recording's real bandwidth.

    A band whose upper edge reaches past the encoder brickwall measures the
    encoder noise floor over part of its width, which biases its level downward
    by an amount that grows with range. Dropping the band is cheaper than
    modelling that bias.

    Args:
        candidate_centre_frequencies_hz: Band centres offered, in hertz.
        codec_cutoff_hz: Measured cutoff of the recording, in hertz.
        codec_cutoff_margin: Fraction of the cutoff a band edge may reach.
        fraction_denominator: Bands per octave, setting the band width.

    Returns:
        The centres whose upper edge stays below the margin, in ascending order.
    """
    half_width_factor = compute_band_half_width_factor(fraction_denominator)
    highest_permitted_edge_hz = codec_cutoff_margin * codec_cutoff_hz
    return tuple(
        centre_hz
        for centre_hz in candidate_centre_frequencies_hz
        if centre_hz * half_width_factor <= highest_permitted_edge_hz
    )


def filter_bands_by_signal_to_noise_ratio(
    band_levels_db: Float64Array,
    noise_floor_levels_db: Float64Array,
    minimum_band_signal_to_noise_ratio_db: float,
) -> BoolArray:
    """Marks the bands whose level stands far enough above the noise floor.

    Args:
        band_levels_db: Measured level per band, in decibels.
        noise_floor_levels_db: Noise floor per band, in decibels.
        minimum_band_signal_to_noise_ratio_db: Smallest usable margin, in decibels.

    Returns:
        Boolean mask over bands, True where the band is usable.

    Raises:
        ValueError: If the two level arrays have different shapes.
    """
    levels_db = np.asarray(band_levels_db, dtype=np.float64)
    floors_db = np.asarray(noise_floor_levels_db, dtype=np.float64)
    if levels_db.shape != floors_db.shape:
        raise ValueError("one noise floor level is required per band level")
    return np.asarray(
        levels_db - floors_db >= minimum_band_signal_to_noise_ratio_db, dtype=np.bool_
    )
