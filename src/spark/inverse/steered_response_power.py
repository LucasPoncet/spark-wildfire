"""Steering the pair correlations over a grid of candidate source positions.

The map is built by pooling each correlation curve over the *interval* of delays
a grid cell spans rather than sampling it at the cell centre. That correction is
load-bearing, not a refinement: the time-difference-of-arrival gradient has
magnitude at most `2 / c`, so a half-metre cell spans up to 2.9 ms of delay,
about 129 samples at 44.1 kHz, while a correlation peak carrying ten kilohertz
of bandwidth is about four samples wide. A point-sampled grid at that spacing
therefore steps over the peak and returns a confident maximum somewhere else,
with nothing in the output to say it happened.
"""

from dataclasses import dataclass

import numpy as np

from src.config.multi_source_localization_configuration import (
    HARMONIC_MEAN_COMBINATOR,
    MAXIMUM_POOLING,
    MEAN_POOLING,
    POINT_POOLING,
    PRODUCT_COMBINATOR,
    SUM_COMBINATOR,
    SUM_POOLING,
    SteeredResponsePowerConfiguration,
)
from src.spark.inverse.time_difference_of_arrival import (
    GeneralizedCrossCorrelationCurve,
)
from src.utils.array_types import Float64Array, Int64Array

MAP_FLOOR: float = 1e-12
POOLING_CHUNK_CELL_COUNT: int = 4096


@dataclass(frozen=True)
class CandidateGrid:
    """A level of the coarse-to-fine search.

    Attributes:
        positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        cell_extent_m: Side length of every cell, in metres.
    """

    positions_xyz_m: Float64Array
    cell_extent_m: float


def build_candidate_grid_xyz(
    domain_extent_x_m: float,
    domain_extent_y_m: float,
    spacing_m: float,
    height_m: float,
) -> CandidateGrid:
    """Tiles the domain with square cells and returns their centres.

    Args:
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        spacing_m: Cell side length, in metres.
        height_m: Height of the candidate plane above the ground, in metres.

    Returns:
        The grid, with x varying fastest.

    Raises:
        ValueError: If the spacing is not positive.
    """
    if spacing_m <= 0.0:
        raise ValueError("candidate grid spacing must be positive")
    x_m = np.arange(0.5 * spacing_m, domain_extent_x_m, spacing_m)
    y_m = np.arange(0.5 * spacing_m, domain_extent_y_m, spacing_m)
    grid_x_m, grid_y_m = np.meshgrid(x_m, y_m)
    positions_xyz_m = np.stack(
        (
            grid_x_m.ravel(),
            grid_y_m.ravel(),
            np.full(grid_x_m.size, height_m, dtype=np.float64),
        ),
        axis=1,
    )
    return CandidateGrid(positions_xyz_m=positions_xyz_m, cell_extent_m=spacing_m)


def subdivide_candidate_cells(
    parent_positions_xyz_m: Float64Array,
    parent_cell_extent_m: float,
    refinement_factor: int,
) -> CandidateGrid:
    """Splits each retained cell into a block of smaller cells.

    Args:
        parent_positions_xyz_m: Centres of the cells to split, shape `(n, 3)`.
        parent_cell_extent_m: Side length of those cells, in metres.
        refinement_factor: Number of children per side.

    Returns:
        The finer grid.

    Raises:
        ValueError: If the refinement factor is smaller than two.
    """
    if refinement_factor < 2:
        raise ValueError("refinement_factor must be at least two")
    parents = np.atleast_2d(np.asarray(parent_positions_xyz_m, dtype=np.float64))
    child_extent_m = parent_cell_extent_m / refinement_factor
    offsets_m = (
        np.arange(refinement_factor, dtype=np.float64) + 0.5
    ) * child_extent_m - 0.5 * parent_cell_extent_m
    offset_x_m, offset_y_m = np.meshgrid(offsets_m, offsets_m)
    displacements_m = np.stack(
        (
            offset_x_m.ravel(),
            offset_y_m.ravel(),
            np.zeros(offset_x_m.size, dtype=np.float64),
        ),
        axis=1,
    )
    children = (parents[:, None, :] + displacements_m[None, :, :]).reshape(-1, 3)
    return CandidateGrid(positions_xyz_m=children, cell_extent_m=child_extent_m)


def compute_pair_delay_table_s(
    candidate_positions_xyz_m: Float64Array,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
) -> Float64Array:
    """Predicts the delay every candidate position would put on every pair.

    Args:
        candidate_positions_xyz_m: Candidate positions, shape `(n_cells, 3)`.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.

    Returns:
        Delays of shape `(n_cells, n_pairs)`, positive where the first receiver
        of the pair is the farther one.
    """
    candidates = np.atleast_2d(np.asarray(candidate_positions_xyz_m, dtype=np.float64))
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    distances_m = np.linalg.norm(candidates[:, None, :] - receivers[None, :, :], axis=2)
    return (
        distances_m[:, pairs[:, 0]] - distances_m[:, pairs[:, 1]]
    ) / speed_of_sound_m_per_s


def compute_distance_bounds_to_cells_m(
    candidate_positions_xyz_m: Float64Array,
    cell_extent_m: float,
    receiver_positions_xyz_m: Float64Array,
) -> tuple[Float64Array, Float64Array]:
    """Bounds the distance from each receiver to anywhere inside each cell.

    Args:
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        cell_extent_m: Cell side length, in metres.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.

    Returns:
        `(nearest_m, farthest_m)`, both of shape `(n_cells, n_receivers)`.
    """
    candidates = np.atleast_2d(np.asarray(candidate_positions_xyz_m, dtype=np.float64))
    receivers = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    half_extent_m = np.array(
        [0.5 * cell_extent_m, 0.5 * cell_extent_m, 0.0], dtype=np.float64
    )
    offsets_m = np.abs(candidates[:, None, :] - receivers[None, :, :])
    nearest_m = np.linalg.norm(np.maximum(offsets_m - half_extent_m, 0.0), axis=2)
    farthest_m = np.linalg.norm(offsets_m + half_extent_m, axis=2)
    return nearest_m, farthest_m


def compute_cell_delay_bounds_s(
    candidate_positions_xyz_m: Float64Array,
    cell_extent_m: float,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
) -> tuple[Float64Array, Float64Array]:
    """Bounds the delay interval each cell spans, on each pair.

    The bounds come from bounding each receiver distance over the cell box and
    differencing the extremes. They are guaranteed to contain every delay the
    cell can produce, which is what lets the search discard a cell without any
    chance of discarding the true maximum with it.

    Args:
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        cell_extent_m: Cell side length, in metres.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.

    Returns:
        `(lower_s, upper_s)`, both of shape `(n_cells, n_pairs)`.
    """
    nearest_m, farthest_m = compute_distance_bounds_to_cells_m(
        candidate_positions_xyz_m, cell_extent_m, receiver_positions_xyz_m
    )
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    lower_s = (
        nearest_m[:, pairs[:, 0]] - farthest_m[:, pairs[:, 1]]
    ) / speed_of_sound_m_per_s
    upper_s = (
        farthest_m[:, pairs[:, 0]] - nearest_m[:, pairs[:, 1]]
    ) / speed_of_sound_m_per_s
    return lower_s, upper_s


def pool_curve_over_intervals(
    curve: GeneralizedCrossCorrelationCurve,
    lower_lags_s: Float64Array,
    upper_lags_s: Float64Array,
    pooling: str,
) -> Float64Array:
    """Reduces a correlation curve over one lag interval per cell.

    Args:
        curve: The correlation curve to read.
        lower_lags_s: Lower bound of each cell interval, shape `(n_cells,)`.
        upper_lags_s: Upper bound of each cell interval, same shape.
        pooling: `sum`, `mean`, `max`, or `point` for the interval midpoint only.

    Returns:
        One value per cell, shape `(n_cells,)`.

    Raises:
        ValueError: If the pooling rule is not recognised.
    """
    if pooling not in (SUM_POOLING, MEAN_POOLING, MAXIMUM_POOLING, POINT_POOLING):
        raise ValueError(f"unknown pooling rule: {pooling}")
    values = np.asarray(curve.values, dtype=np.float64)
    first_lag_s = float(curve.lags_s[0])
    lower_indices = np.clip(
        np.floor((np.asarray(lower_lags_s) - first_lag_s) * curve.sample_rate_hz),
        0,
        values.size - 1,
    ).astype(np.int64)
    upper_indices = np.clip(
        np.ceil((np.asarray(upper_lags_s) - first_lag_s) * curve.sample_rate_hz),
        0,
        values.size - 1,
    ).astype(np.int64)

    if pooling == POINT_POOLING:
        centre_indices = (lower_indices + upper_indices) // 2
        return np.asarray(values[centre_indices], dtype=np.float64)

    widths = upper_indices - lower_indices + 1
    maximum_width = int(np.max(widths))
    pooled = np.empty(lower_indices.size, dtype=np.float64)
    offsets = np.arange(maximum_width, dtype=np.int64)
    for start in range(0, lower_indices.size, POOLING_CHUNK_CELL_COUNT):
        stop = min(start + POOLING_CHUNK_CELL_COUNT, lower_indices.size)
        chunk_lower = lower_indices[start:stop, None]
        chunk_widths = widths[start:stop, None]
        gathered_indices = np.minimum(
            chunk_lower + offsets[None, :], upper_indices[start:stop, None]
        )
        gathered = values[gathered_indices]
        inside = offsets[None, :] < chunk_widths
        if pooling == MAXIMUM_POOLING:
            pooled[start:stop] = np.max(np.where(inside, gathered, -np.inf), axis=1)
        else:
            totals = np.sum(np.where(inside, gathered, 0.0), axis=1)
            pooled[start:stop] = (
                totals / chunk_widths[:, 0] if pooling == MEAN_POOLING else totals
            )
    return pooled


def combine_pairwise_maps(
    pairwise_maps: Float64Array, pairwise_combinator: str
) -> Float64Array:
    """Merges one map per receiver pair into one map over the grid.

    Summing lets a single pair with a strong spurious peak carry the global map.
    The product instead requires every pair to agree, and the harmonic mean sits
    between the two while suppressing sidelobes.

    The product is returned as its geometric mean. Taking the root is monotone,
    so it moves no maximum, but it keeps the map on the same scale as one
    pairwise map however many pairs there are. Without it the raw product of
    fifteen maps runs to `1e-180` and every ratio taken against the map median
    stops meaning anything.

    Args:
        pairwise_maps: Maps of shape `(n_pairs, n_cells)`, each already
            non-negative.
        pairwise_combinator: `sum`, `product` or `harmonic_mean`.

    Returns:
        The combined map, shape `(n_cells,)`.

    Raises:
        ValueError: If the combinator is not recognised.
    """
    maps = np.atleast_2d(np.asarray(pairwise_maps, dtype=np.float64))
    normalised = maps - np.min(maps, axis=1, keepdims=True)
    normalised = normalised / (np.max(normalised, axis=1, keepdims=True) + MAP_FLOOR)
    if pairwise_combinator == SUM_COMBINATOR:
        return np.asarray(np.sum(normalised, axis=0), dtype=np.float64)
    if pairwise_combinator == PRODUCT_COMBINATOR:
        return np.asarray(
            np.exp(np.mean(np.log(normalised + MAP_FLOOR), axis=0)), dtype=np.float64
        )
    if pairwise_combinator == HARMONIC_MEAN_COMBINATOR:
        return np.asarray(
            maps.shape[0] / np.sum(1.0 / (normalised + MAP_FLOOR), axis=0),
            dtype=np.float64,
        )
    raise ValueError(f"unknown pairwise combinator: {pairwise_combinator}")


def compute_steered_response_power_map(
    correlation_curves: list[GeneralizedCrossCorrelationCurve],
    delay_lower_bounds_s: Float64Array,
    delay_upper_bounds_s: Float64Array,
    pooling: str,
    pairwise_combinator: str,
) -> Float64Array:
    """Builds the steered response power over a candidate grid.

    Args:
        correlation_curves: One curve per receiver pair.
        delay_lower_bounds_s: Lower delay bound, shape `(n_cells, n_pairs)`.
        delay_upper_bounds_s: Upper delay bound, same shape.
        pooling: How a cell interval is reduced to one value.
        pairwise_combinator: How the pairwise maps are merged.

    Returns:
        The map, shape `(n_cells,)`.

    Raises:
        ValueError: If the bound arrays do not carry one column per curve.
    """
    lower_s = np.atleast_2d(np.asarray(delay_lower_bounds_s, dtype=np.float64))
    upper_s = np.atleast_2d(np.asarray(delay_upper_bounds_s, dtype=np.float64))
    if lower_s.shape[1] != len(correlation_curves):
        raise ValueError("one column of delay bounds is required per correlation curve")
    pairwise_maps = np.stack(
        [
            pool_curve_over_intervals(
                curve, lower_s[:, pair_index], upper_s[:, pair_index], pooling
            )
            for pair_index, curve in enumerate(correlation_curves)
        ]
    )
    return combine_pairwise_maps(pairwise_maps, pairwise_combinator)


def extract_map_peak(
    steered_response_power_map: Float64Array,
    candidate_positions_xyz_m: Float64Array,
) -> tuple[Float64Array, float, float]:
    """Reads the strongest candidate off a map.

    Args:
        steered_response_power_map: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.

    Returns:
        `(position_xyz_m, peak_value, peak_to_median_ratio)`.
    """
    power_map = np.asarray(steered_response_power_map, dtype=np.float64)
    candidates = np.atleast_2d(np.asarray(candidate_positions_xyz_m, dtype=np.float64))
    peak_index = int(np.argmax(power_map))
    peak_value = float(power_map[peak_index])
    median_value = float(np.median(power_map))
    return (
        np.asarray(candidates[peak_index], dtype=np.float64),
        peak_value,
        peak_value / (abs(median_value) + MAP_FLOOR),
    )


def select_retained_cells(
    steered_response_power_map: Float64Array,
    candidate_positions_xyz_m: Float64Array,
    retained_fraction: float,
) -> Float64Array:
    """Keeps the highest-scoring cells for the next refinement level.

    Args:
        steered_response_power_map: The map, shape `(n_cells,)`.
        candidate_positions_xyz_m: Cell centres, shape `(n_cells, 3)`.
        retained_fraction: Fraction of cells to carry forward.

    Returns:
        Centres of the retained cells, shape `(n_retained, 3)`.

    Raises:
        ValueError: If the fraction is outside `(0, 1]`.
    """
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must lie in (0, 1]")
    power_map = np.asarray(steered_response_power_map, dtype=np.float64)
    candidates = np.atleast_2d(np.asarray(candidate_positions_xyz_m, dtype=np.float64))
    retained_count = max(round(retained_fraction * power_map.size), 1)
    retained_indices = np.argpartition(power_map, -retained_count)[-retained_count:]
    return np.asarray(candidates[retained_indices], dtype=np.float64)


@dataclass(frozen=True)
class SteeredResponseSearch:
    """The outcome of one coarse-to-fine search over the domain.

    Attributes:
        position_xyz_m: Strongest candidate at the finest level, shape `(3,)`.
        peak_value: Map value there.
        coarse_peak_to_median_ratio: Coarse map peak over its median. The
            whole-domain map is where a source either stands out or does not, so
            this rather than the refined peak is what decides whether a source
            is there at all.
        final_cell_extent_m: Cell side length at the finest level, in metres.
        coarse_map: The first level map, kept for display, shape `(n_cells,)`.
        coarse_positions_xyz_m: Cell centres of that level, shape `(n_cells, 3)`.
    """

    position_xyz_m: Float64Array
    peak_value: float
    coarse_peak_to_median_ratio: float
    final_cell_extent_m: float
    coarse_map: Float64Array
    coarse_positions_xyz_m: Float64Array


def compute_map_over_grid(
    correlation_curves: list[GeneralizedCrossCorrelationCurve],
    grid: CandidateGrid,
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    configuration: SteeredResponsePowerConfiguration,
) -> Float64Array:
    """Bounds every cell delay and pools the curves over those bounds.

    Args:
        correlation_curves: One curve per receiver pair.
        grid: The candidate grid level to evaluate.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        configuration: Pooling and combinator settings.

    Returns:
        The map over the grid, shape `(n_cells,)`.
    """
    lower_s, upper_s = compute_cell_delay_bounds_s(
        grid.positions_xyz_m,
        grid.cell_extent_m,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
    )
    return compute_steered_response_power_map(
        correlation_curves,
        lower_s,
        upper_s,
        configuration.pooling,
        configuration.pairwise_combinator,
    )


def run_coarse_to_fine_search(
    seed_correlation_curves: list[GeneralizedCrossCorrelationCurve],
    correlation_curves: list[GeneralizedCrossCorrelationCurve],
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    domain_extent_x_m: float,
    domain_extent_y_m: float,
    configuration: SteeredResponsePowerConfiguration,
) -> SteeredResponseSearch:
    """Searches the whole domain coarsely, then refines around what survived.

    The first level is evaluated on curves restricted to low frequencies. Peak
    width goes inversely with frequency, so that map is deliberately smooth and
    its basins are wide enough that a coarse cell cannot fall between two of
    them; the finer levels then use the full band.

    Args:
        seed_correlation_curves: Low-frequency curves for the first level.
        correlation_curves: Full-band curves for the finer levels.
        receiver_positions_xyz_m: Receiver positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        domain_extent_x_m: Domain extent along x, in metres.
        domain_extent_y_m: Domain extent along y, in metres.
        configuration: Pooling, combinator, spacing and refinement settings.

    Returns:
        The strongest candidate with its prominence and the coarse map.
    """
    grid = build_candidate_grid_xyz(
        domain_extent_x_m,
        domain_extent_y_m,
        configuration.coarse_grid_spacing_m,
        configuration.candidate_height_m,
    )
    coarse_positions_xyz_m = grid.positions_xyz_m
    coarse_map = compute_map_over_grid(
        seed_correlation_curves,
        grid,
        receiver_positions_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
        configuration,
    )
    position_xyz_m, peak_value, coarse_peak_to_median_ratio = extract_map_peak(
        coarse_map, coarse_positions_xyz_m
    )

    level_map = coarse_map
    for _ in range(configuration.refinement_levels):
        grid = subdivide_candidate_cells(
            select_retained_cells(
                level_map,
                grid.positions_xyz_m,
                configuration.refinement_retained_fraction,
            ),
            grid.cell_extent_m,
            configuration.refinement_factor,
        )
        level_map = compute_map_over_grid(
            correlation_curves,
            grid,
            receiver_positions_xyz_m,
            receiver_pairs,
            speed_of_sound_m_per_s,
            configuration,
        )
        position_xyz_m, peak_value, _ = extract_map_peak(
            level_map, grid.positions_xyz_m
        )

    return SteeredResponseSearch(
        position_xyz_m=position_xyz_m,
        peak_value=peak_value,
        coarse_peak_to_median_ratio=coarse_peak_to_median_ratio,
        final_cell_extent_m=grid.cell_extent_m,
        coarse_map=coarse_map,
        coarse_positions_xyz_m=coarse_positions_xyz_m,
    )
