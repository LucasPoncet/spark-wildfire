"""Receiver position layouts. Pure geometry, no acoustics.

A layout is an (n_receivers, 2) array of xy positions in metres. Receivers
sit on the ground plane, so z is not carried here.
"""

import numpy as np

from src.config.receiver_configuration import (
    GRID_PLACEMENT,
    RANDOM_PLACEMENT,
    RING_PLACEMENT,
    ReceiverConfiguration,
)
from src.utils.array_types import Float64Array

REJECTION_SAMPLING_BATCH_FACTOR: int = 4


def place_receivers_in_ring(
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    receiver_count: int,
) -> Float64Array:
    """Space receivers evenly around a circle, starting on the positive x-axis.

    Args:
        center_x_m: Ring centre along x, in metres.
        center_y_m: Ring centre along y, in metres.
        radius_m: Ring radius, in metres.
        receiver_count: Number of receivers to place.

    Returns:
        Float64 array of shape (receiver_count, 2).

    Raises:
        ValueError: If the receiver count is not positive.
    """
    if receiver_count <= 0:
        raise ValueError("receiver count must be positive")

    angles_rad = np.linspace(0.0, 2.0 * np.pi, receiver_count, endpoint=False)
    return np.stack(
        (
            center_x_m + radius_m * np.cos(angles_rad),
            center_y_m + radius_m * np.sin(angles_rad),
        ),
        axis=1,
    )


def place_receivers_in_grid(
    origin_x_m: float,
    origin_y_m: float,
    extent_x_m: float,
    extent_y_m: float,
    spacing_m: float,
) -> Float64Array:
    """Tile receivers on a regular grid covering the extent, endpoints included.

    Args:
        origin_x_m: Lower-left corner along x, in metres.
        origin_y_m: Lower-left corner along y, in metres.
        extent_x_m: Width covered, in metres.
        extent_y_m: Height covered, in metres.
        spacing_m: Distance between adjacent receivers, in metres.

    Returns:
        Float64 array of shape (n_receivers, 2), x varying fastest.

    Raises:
        ValueError: If the spacing is not positive.
    """
    if spacing_m <= 0.0:
        raise ValueError("receiver spacing must be positive")

    half_spacing_m = 0.5 * spacing_m
    x_m = np.arange(origin_x_m, origin_x_m + extent_x_m + half_spacing_m, spacing_m)
    y_m = np.arange(origin_y_m, origin_y_m + extent_y_m + half_spacing_m, spacing_m)
    grid_x_m, grid_y_m = np.meshgrid(x_m, y_m)
    return np.stack((grid_x_m.ravel(), grid_y_m.ravel()), axis=1)


def place_receivers_randomly(
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    receiver_count: int,
    seed: int = 0,
) -> Float64Array:
    """Draw receivers uniformly inside a disc by rejection sampling.

    Candidates are drawn in vectorized batches from the bounding square and
    the ones outside the disc are discarded, repeating until enough survive.

    Args:
        center_x_m: Disc centre along x, in metres.
        center_y_m: Disc centre along y, in metres.
        radius_m: Disc radius, in metres.
        receiver_count: Number of receivers to place.
        seed: Seed making the layout reproducible.

    Returns:
        Float64 array of shape (receiver_count, 2).

    Raises:
        ValueError: If the receiver count or the radius is not positive.
    """
    if receiver_count <= 0:
        raise ValueError("receiver count must be positive")
    if radius_m <= 0.0:
        raise ValueError("sampling radius must be positive")

    generator = np.random.default_rng(seed)
    batch_size = REJECTION_SAMPLING_BATCH_FACTOR * receiver_count
    accepted_offsets_xy_m = np.empty((0, 2), dtype=np.float64)
    while accepted_offsets_xy_m.shape[0] < receiver_count:
        candidates_xy_m = generator.uniform(-radius_m, radius_m, size=(batch_size, 2))
        is_inside_disc = np.sum(candidates_xy_m**2, axis=1) <= radius_m**2
        accepted_offsets_xy_m = np.concatenate(
            (accepted_offsets_xy_m, candidates_xy_m[is_inside_disc])
        )
    return accepted_offsets_xy_m[:receiver_count] + np.array(
        [center_x_m, center_y_m], dtype=np.float64
    )


def place_receivers_from_configuration(
    config: ReceiverConfiguration,
    grid_extent_x_m: float,
    grid_extent_y_m: float,
) -> Float64Array:
    """Build the layout the configuration names, resolving fractional centres.

    Args:
        config: Layout strategy and its parameters.
        grid_extent_x_m: Domain extent along x, in metres.
        grid_extent_y_m: Domain extent along y, in metres.

    Returns:
        Float64 array of shape (n_receivers, 2).

    Raises:
        ValueError: If the placement strategy is not recognised.
    """
    center_x_m = config.ring_center_x_fraction * grid_extent_x_m
    center_y_m = config.ring_center_y_fraction * grid_extent_y_m

    if config.placement_strategy == RING_PLACEMENT:
        return place_receivers_in_ring(
            center_x_m, center_y_m, config.ring_radius_m, config.receiver_count
        )
    if config.placement_strategy == GRID_PLACEMENT:
        return place_receivers_in_grid(
            0.0, 0.0, grid_extent_x_m, grid_extent_y_m, config.grid_spacing_m
        )
    if config.placement_strategy == RANDOM_PLACEMENT:
        return place_receivers_randomly(
            center_x_m,
            center_y_m,
            config.random_radius_m,
            config.receiver_count,
            config.random_seed,
        )
    raise ValueError(
        f"unknown receiver placement strategy: {config.placement_strategy}"
    )
