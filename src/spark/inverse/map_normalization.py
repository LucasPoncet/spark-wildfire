"""Putting a steered response power map on a scale that can be compared.

Absolute map values are not comparable across time. The number of radiating
cells in a spreading fire grows by more than an order of magnitude over one
run, and the pairwise combinator rescales every pair map before merging it, so
a frame's peak height says as much about how many cells were alight as about
where they were. Every cross-frame quantity therefore runs on a prepared map:
background removed, total mass one.

Nothing here knows what produced the map, which is what lets the analytic point
spread function and a rendered frame go through the identical path. That
identity is load-bearing rather than tidy: the covariance subtraction in the
extent stage differences two second moments, so any preprocessing applied to
one and not the other appears as a bias in the difference.
"""

import numpy as np

from src.utils.array_types import Float64Array

MASS_FLOOR: float = 1e-300


def subtract_map_background(
    map_values: Float64Array, background_percentile: float
) -> Float64Array:
    """Removes the map's own background level and clips what is left at zero.

    The sidelobe floor of a steered response power map is broad and carries
    most of the grid, so a moment taken over the raw map is dominated by the
    domain's geometry rather than by the source. Subtracting a percentile of
    the frame removes it without assuming a noise model.

    Applying this twice changes nothing: after one pass the bottom
    `background_percentile` per cent of cells are exactly zero, so the same
    percentile of the result is itself zero.

    Args:
        map_values: The map, shape `(n_cells,)`.
        background_percentile: Percentile of the frame treated as background,
            between zero and one hundred.

    Returns:
        The map with its background removed, shape `(n_cells,)`, non-negative.

    Raises:
        ValueError: If the percentile is outside `[0, 100]`.
    """
    if not 0.0 <= background_percentile <= 100.0:
        raise ValueError("background percentile must lie between 0 and 100")
    values = np.asarray(map_values, dtype=np.float64)
    background = float(np.percentile(values, background_percentile))
    return np.asarray(np.clip(values - background, 0.0, None), dtype=np.float64)


def normalize_map_to_unit_mass(map_values: Float64Array) -> Float64Array:
    """Scales a non-negative map so its cells sum to one.

    Args:
        map_values: The map, shape `(n_cells,)`, non-negative.

    Returns:
        The map scaled to unit total mass, shape `(n_cells,)`. An all-zero map
        is returned unchanged rather than divided by zero, so a frame in which
        nothing radiated stays empty instead of becoming uniform.
    """
    values = np.asarray(map_values, dtype=np.float64)
    total_mass = float(np.sum(values))
    if total_mass <= MASS_FLOOR:
        return values
    return np.asarray(values / total_mass, dtype=np.float64)


def prepare_map_for_moments(
    map_values: Float64Array, background_percentile: float
) -> Float64Array:
    """The one preprocessing path every moment in this package runs through.

    Args:
        map_values: The map, shape `(n_cells,)`.
        background_percentile: Percentile of the frame treated as background.

    Returns:
        Background-subtracted, unit-mass map of shape `(n_cells,)`.
    """
    return normalize_map_to_unit_mass(
        subtract_map_background(map_values, background_percentile)
    )
