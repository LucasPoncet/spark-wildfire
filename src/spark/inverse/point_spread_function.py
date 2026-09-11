"""What one point source looks like after the array has imaged it.

The response is fixed by array geometry, the whitening band, the speed of
sound and the pooling scheme. No forward model is involved and no rendering is
required, which is why this belongs in `inverse/` and works unchanged on
recorded data.

The map of a point source at `u0` is the pairwise combination of each pair's
correlation curve read at the delay error `tau_ij(u) - tau_ij(u0)`. Rather than
evaluating that by hand, this module synthesises the cross-spectrum a noiseless
point source would produce and hands it to the same curve builder and the same
grid evaluator a rendered frame goes through. The alternative — an analytic
expression for the lobe — silently omits the analytic envelope, the lag
truncation and the per-pair rescaling inside the combinator, and the extent
stage subtracts one covariance from the other, so any such omission comes back
as a bias rather than as a visible discrepancy.

`compute_whitened_autocorrelation_kernel` gives the closed form anyway, because
a one-dimensional profile is what the radial front model needs and because
agreement between it and the synthesised curve is a cheap check on both.

**Each pair's curve is scaled by the levels that pair would actually receive.**
Under a combinator that rescales every pair map before merging, pair amplitude
cancels and this would be wasted work. Under `unscaled_sum` — which the imaging
family needs, because rescaling is what stops a map adding — it does not: pairs
contribute in proportion to what they hear, so near pairs dominate and the
response is not the one an equally-weighted model predicts. Leaving it out cost
12 to 16 per cent on the response semi-axis away from the array centre, and no
amount of grid refinement touched it.

The scale is free-field spreading computed from the receiver positions, which
is geometry the estimator assumes rather than anything read back from a forward
model; `inverse/` still imports nothing from `acoustic/`.
"""

import numpy as np

from src.config.multi_source_localization_configuration import (
    CorrelationConfiguration,
    SteeredResponsePowerConfiguration,
)
from src.spark.inverse.map_moments import compute_map_moments
from src.spark.inverse.steered_response_power import (
    CandidateGrid,
    compute_map_over_grid,
    compute_pair_delay_table_s,
)
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
    build_correlation_curve_values,
    compute_band_mask,
)
from src.utils.array_types import BoolArray, ComplexArray, Float64Array, Int64Array

TRANSFORM_LENGTH_MARGIN_SAMPLES: int = 4
DISTANCE_PRODUCT_FLOOR_M2: float = 1e-6


def compute_whitened_autocorrelation_kernel(
    lag_values_s: Float64Array,
    lowest_frequency_hz: float,
    highest_frequency_hz: float,
    phase_transform_exponent: float,
) -> Float64Array:
    """Closed-form correlation lobe of a point source with flat in-band content.

    Fully whitening a band-limited spectrum leaves unit magnitude between the
    band edges, whose inverse transform is a difference of two sincs, unit at
    zero lag.

    The exponent is accepted and does not change the result, which is the
    honest statement rather than an omission: the phase transform divides by
    the magnitude of the mixture, so with flat in-band content
    `|X|^(1 - beta)` is constant across the band whatever `beta` is. The
    exponent therefore acts only through spectral non-flatness, which is a
    property of the sources present and not of the array. Where that matters,
    pass a measured magnitude to `build_point_spread_correlation_curves`.

    Args:
        lag_values_s: Lags to evaluate at, shape `(n_lags,)`.
        lowest_frequency_hz: Lower edge of the whitening band, in hertz.
        highest_frequency_hz: Upper edge of that band, in hertz.
        phase_transform_exponent: The `beta` of the phase transform.

    Returns:
        The lobe at each lag, shape `(n_lags,)`, peaking at one.

    Raises:
        ValueError: If the band is empty or inverted.
    """
    if highest_frequency_hz <= lowest_frequency_hz:
        raise ValueError("the whitening band must have positive width")
    del phase_transform_exponent
    lags_s = np.asarray(lag_values_s, dtype=np.float64)
    bandwidth_hz = highest_frequency_hz - lowest_frequency_hz
    return np.asarray(
        (
            highest_frequency_hz * np.sinc(2.0 * highest_frequency_hz * lags_s)
            - lowest_frequency_hz * np.sinc(2.0 * lowest_frequency_hz * lags_s)
        )
        / bandwidth_hz,
        dtype=np.float64,
    )


def build_point_spread_correlation_curves(
    reference_position_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    sample_rate_hz: int,
    maximum_absolute_lag_s: float,
    correlation_configuration: CorrelationConfiguration,
    band_magnitude_spectrum: Float64Array | None,
) -> list[GeneralizedCrossCorrelationCurve]:
    """Synthesises the curves a noiseless point source would produce.

    Args:
        reference_position_xyz_m: Where the point source sits, shape `(3,)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        sample_rate_hz: Sample rate in hertz.
        maximum_absolute_lag_s: Largest lag to retain, in seconds.
        correlation_configuration: Whitening band, exponent and envelope, taken
            from the same configuration a real frame is built with.
        band_magnitude_spectrum: Cross-spectrum magnitude per retained bin,
            shape `(n_retained_bins,)`, or None for flat in-band content. None
            is the fully whitened case and is exact whenever the exponent is
            one; pass a measured magnitude when the exponent is below one and
            the content is not flat.

    Returns:
        One curve per pair, in the order the pairs are given.
    """
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    reference_delays_s = compute_pair_delay_table_s(
        np.asarray(reference_position_xyz_m, dtype=np.float64).reshape(1, 3),
        receivers,
        pairs,
        speed_of_sound_m_per_s,
    )[0]

    reference_distances_m = np.linalg.norm(
        receivers - np.asarray(reference_position_xyz_m, dtype=np.float64), axis=1
    )
    pair_scales = compute_pair_amplitude_scales(reference_distances_m, pairs)

    maximum_shift = round(maximum_absolute_lag_s * sample_rate_hz)
    transform_length = 2 * (maximum_shift + TRANSFORM_LENGTH_MARGIN_SAMPLES)
    frequencies_hz = np.fft.rfftfreq(transform_length, 1.0 / sample_rate_hz)
    band_mask = compute_band_mask(
        frequencies_hz,
        correlation_configuration.lowest_frequency_hz,
        correlation_configuration.highest_frequency_hz,
    )
    magnitude = _resolve_band_magnitude(band_mask, band_magnitude_spectrum)

    curves: list[GeneralizedCrossCorrelationCurve] = []
    for pair_index in range(pairs.shape[0]):
        spectrum: ComplexArray = np.asarray(
            float(pair_scales[pair_index])
            * magnitude
            * np.exp(
                -2j * np.pi * frequencies_hz * float(reference_delays_s[pair_index])
            ),
            dtype=np.complex128,
        )
        lags_s, values = build_correlation_curve_values(
            spectrum,
            transform_length,
            sample_rate_hz,
            maximum_absolute_lag_s,
            correlation_configuration.use_analytic_envelope,
        )
        curves.append(
            GeneralizedCrossCorrelationCurve(
                receiver_pair=(int(pairs[pair_index, 0]), int(pairs[pair_index, 1])),
                lags_s=lags_s,
                values=values,
                sample_rate_hz=sample_rate_hz,
                window_count=1,
                effective_bandwidth_hz=float(
                    np.sqrt(np.mean(frequencies_hz[band_mask] ** 2))
                )
                if np.any(band_mask)
                else 0.0,
            )
        )
    return curves


def compute_pair_amplitude_scales(
    reference_distances_m: Float64Array, receiver_pairs: Int64Array
) -> Float64Array:
    """How loudly each pair hears a source at the reference position.

    A cross-spectrum is one channel times the conjugate of another, so its
    magnitude carries both receivers' amplitudes. Under free-field spreading
    each is inversely proportional to its own range, making the pair's scale
    the product of two reciprocals.

    Args:
        reference_distances_m: Range from the source to each receiver, shape
            `(n_receivers,)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.

    Returns:
        One scale per pair, shape `(n_pairs,)`, normalised so the loudest pair
        is one. The normalisation keeps the synthesised curves on the same
        order as measured ones without changing their relative weighting.
    """
    distances_m = np.asarray(reference_distances_m, dtype=np.float64)
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    scales = 1.0 / np.maximum(
        distances_m[pairs[:, 0]] * distances_m[pairs[:, 1]], DISTANCE_PRODUCT_FLOOR_M2
    )
    largest = float(np.max(scales))
    return np.asarray(scales / largest if largest > 0.0 else scales, dtype=np.float64)


def compute_point_spread_function(
    reference_position_xyz_m: Float64Array,
    grid: CandidateGrid,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    sample_rate_hz: int,
    maximum_absolute_lag_s: float,
    correlation_configuration: CorrelationConfiguration,
    steered_response_power_configuration: SteeredResponsePowerConfiguration,
    band_magnitude_spectrum: Float64Array | None,
) -> Float64Array:
    """The imaging map a single point source at the reference position produces.

    Args:
        reference_position_xyz_m: Where the point source sits, shape `(3,)`.
        grid: The imaging grid, identical to the one frames are evaluated on.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        sample_rate_hz: Sample rate in hertz.
        maximum_absolute_lag_s: Largest lag retained, in seconds.
        correlation_configuration: Whitening band, exponent and envelope.
        steered_response_power_configuration: Pooling and combinator.
        band_magnitude_spectrum: Measured magnitude per retained bin, or None
            for flat in-band content.

    Returns:
        The response over the grid, shape `(n_cells,)`.
    """
    return compute_map_over_grid(
        build_point_spread_correlation_curves(
            reference_position_xyz_m,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            sample_rate_hz,
            maximum_absolute_lag_s,
            correlation_configuration,
            band_magnitude_spectrum,
        ),
        grid,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
        steered_response_power_configuration,
    )


def compute_point_spread_covariance(
    reference_position_xyz_m: Float64Array,
    grid: CandidateGrid,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    sample_rate_hz: int,
    maximum_absolute_lag_s: float,
    correlation_configuration: CorrelationConfiguration,
    steered_response_power_configuration: SteeredResponsePowerConfiguration,
    band_magnitude_spectrum: Float64Array | None,
    background_percentile: float,
) -> Float64Array:
    """Second central moment of the response to a point source.

    Runs through `compute_map_moments`, which is the same preparation a real
    frame gets, because the extent stage subtracts this from an observed
    covariance and a difference of differently-prepared moments is a bias.

    Args:
        reference_position_xyz_m: Where the point source sits, shape `(3,)`.
        grid: The imaging grid.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        sample_rate_hz: Sample rate in hertz.
        maximum_absolute_lag_s: Largest lag retained, in seconds.
        correlation_configuration: Whitening band, exponent and envelope.
        steered_response_power_configuration: Pooling and combinator.
        band_magnitude_spectrum: Measured magnitude per bin, or None.
        background_percentile: Percentile of the frame treated as background.

    Returns:
        The response covariance in metres squared, shape `(2, 2)`.
    """
    response = compute_point_spread_function(
        reference_position_xyz_m,
        grid,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
        sample_rate_hz,
        maximum_absolute_lag_s,
        correlation_configuration,
        steered_response_power_configuration,
        band_magnitude_spectrum,
    )
    return compute_map_moments(
        response, grid.positions_xyz_m, background_percentile
    ).covariance_xy_m2


def build_point_spread_covariance_field(
    sample_positions_xyz_m: Float64Array,
    grid: CandidateGrid,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    sample_rate_hz: int,
    maximum_absolute_lag_s: float,
    correlation_configuration: CorrelationConfiguration,
    steered_response_power_configuration: SteeredResponsePowerConfiguration,
    band_magnitude_spectrum: Float64Array | None,
    background_percentile: float,
) -> tuple[Float64Array, Float64Array]:
    """Tabulates the response covariance across the domain.

    A ring array's response is close to isotropic at the centre and both wider
    and skewed towards the edge, so the covariance to subtract depends on where
    the centroid sits. Evaluating it on a coarse field once and interpolating
    is what keeps the extent stage from recomputing a response per frame.

    Args:
        sample_positions_xyz_m: Nodes to evaluate at, shape `(n_nodes, 3)`.
        grid: The imaging grid.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        sample_rate_hz: Sample rate in hertz.
        maximum_absolute_lag_s: Largest lag retained, in seconds.
        correlation_configuration: Whitening band, exponent and envelope.
        steered_response_power_configuration: Pooling and combinator.
        band_magnitude_spectrum: Measured magnitude per bin, or None.
        background_percentile: Percentile of the frame treated as background.

    Returns:
        `(node_positions_xyz_m, covariances_m2)` of shapes `(n_nodes, 3)` and
        `(n_nodes, 2, 2)`.
    """
    nodes = np.atleast_2d(np.asarray(sample_positions_xyz_m, dtype=np.float64))
    covariances_m2 = np.empty((nodes.shape[0], 2, 2), dtype=np.float64)
    for node_index in range(nodes.shape[0]):
        covariances_m2[node_index] = compute_point_spread_covariance(
            nodes[node_index],
            grid,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            sample_rate_hz,
            maximum_absolute_lag_s,
            correlation_configuration,
            steered_response_power_configuration,
            band_magnitude_spectrum,
            background_percentile,
        )
    return nodes, covariances_m2


def compute_point_spread_semi_axes_m(covariance_m2: Float64Array) -> Float64Array:
    """Major and minor one-sigma semi-axes of a response covariance.

    Args:
        covariance_m2: Response covariance, shape `(2, 2)`.

    Returns:
        `(major_m, minor_m)`, shape `(2,)`, descending.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(covariance_m2, dtype=np.float64))
    return np.asarray(np.sqrt(np.clip(eigenvalues, 0.0, None))[::-1], dtype=np.float64)


def _resolve_band_magnitude(
    band_mask: BoolArray, band_magnitude_spectrum: Float64Array | None
) -> Float64Array:
    """Places the retained magnitudes into a full-length spectrum of zeros.

    Args:
        band_mask: Which bins the whitening retains, shape `(n_bins,)`.
        band_magnitude_spectrum: Magnitude per retained bin, or None for flat.

    Returns:
        Magnitude over every bin, shape `(n_bins,)`.

    Raises:
        ValueError: If a supplied magnitude does not match the retained bins.
    """
    mask = np.asarray(band_mask, dtype=np.bool_)
    magnitude = np.zeros(mask.size, dtype=np.float64)
    if band_magnitude_spectrum is None:
        magnitude[mask] = 1.0
        return magnitude
    supplied = np.asarray(band_magnitude_spectrum, dtype=np.float64)
    retained_count = int(np.count_nonzero(mask))
    if supplied.size != retained_count:
        raise ValueError(
            f"the supplied magnitude has {supplied.size} bins but the whitening "
            f"band retains {retained_count}"
        )
    magnitude[mask] = supplied
    return magnitude
