"""How far each map family is from adding the way imaging assumes.

One panel per source count, the normalised residual against the phase
transform exponent, one line per combinator and pooling rule. The acceptance
target is drawn as a horizontal rule so a family that misses it is visible as
such rather than having to be read off the axis.

The residual is the quantity plotted rather than the correlation, because
correlation saturates near one long before a map is additive enough to
deconvolve, and a figure that shows every family at 0.99 would suggest the
choice does not matter.
"""

import numpy as np
from matplotlib.figure import Figure

from src.utils.array_types import Float64Array

SERIES_COLORS: tuple[str, ...] = ("#2f5d8a", "#d94a3d", "#43a047", "#8e5fb0")
TARGET_COLOR: str = "#888888"
FIGURE_HEIGHT_INCHES: float = 3.6
PANEL_WIDTH_INCHES: float = 3.4


def build_series_label(pairwise_combinator: str, pooling: str) -> str:
    """Names one line of the figure.

    Args:
        pairwise_combinator: How pairwise maps were merged.
        pooling: Which interval pooling rule was applied.

    Returns:
        The label.
    """
    return f"{pairwise_combinator}, {pooling}"


def plot_map_linearity_audit(
    source_counts: list[int],
    phase_transform_exponents: list[float],
    pairwise_combinators: list[str],
    poolings: list[str],
    normalized_residuals: list[float],
    correlations: list[float],
    residual_target: float,
) -> Figure:
    """Draws the audit as residual against exponent, one panel per source count.

    Args:
        source_counts: Source count of each measured cell.
        phase_transform_exponents: Exponent of each cell.
        pairwise_combinators: Combinator of each cell.
        poolings: Pooling rule of each cell.
        normalized_residuals: Residual of each cell.
        correlations: Correlation of each cell, annotated on the best point.
        residual_target: Acceptance threshold, drawn as a rule.

    Returns:
        A matplotlib Figure. Callers save it; this function never does.

    Raises:
        ValueError: If the columns are not all the same length.
    """
    lengths = {
        len(source_counts),
        len(phase_transform_exponents),
        len(pairwise_combinators),
        len(poolings),
        len(normalized_residuals),
        len(correlations),
    }
    if len(lengths) != 1:
        raise ValueError("every audit column must have the same length")

    counts = sorted(set(source_counts))
    series = sorted(
        {
            build_series_label(combinator, pooling)
            for combinator, pooling in zip(pairwise_combinators, poolings, strict=True)
        }
    )
    figure = Figure(
        figsize=(PANEL_WIDTH_INCHES * max(1, len(counts)), FIGURE_HEIGHT_INCHES)
    )
    for panel_index, source_count in enumerate(counts):
        axes = figure.add_subplot(1, len(counts), panel_index + 1)
        for series_index, label in enumerate(series):
            exponents, residuals = _select_series(
                source_counts,
                phase_transform_exponents,
                pairwise_combinators,
                poolings,
                normalized_residuals,
                source_count,
                label,
            )
            if exponents.size == 0:
                continue
            axes.plot(
                exponents,
                residuals,
                marker="o",
                markersize=4,
                color=SERIES_COLORS[series_index % len(SERIES_COLORS)],
                label=label,
            )
        axes.axhline(residual_target, color=TARGET_COLOR, linestyle="--", linewidth=1.0)
        axes.set_yscale("log")
        axes.set_xlabel("phase transform exponent")
        axes.set_title(f"{source_count} sources")
        if panel_index == 0:
            axes.set_ylabel("normalised residual")
            axes.legend(fontsize=7, framealpha=0.8)
    figure.suptitle("map linearity: joint map against the sum of individual maps")
    figure.tight_layout()
    return figure


def _select_series(
    source_counts: list[int],
    phase_transform_exponents: list[float],
    pairwise_combinators: list[str],
    poolings: list[str],
    normalized_residuals: list[float],
    source_count: int,
    label: str,
) -> tuple[Float64Array, Float64Array]:
    """Pulls one line out of the flat cell table, ordered by exponent.

    Args:
        source_counts: Source count of each cell.
        phase_transform_exponents: Exponent of each cell.
        pairwise_combinators: Combinator of each cell.
        poolings: Pooling rule of each cell.
        normalized_residuals: Residual of each cell.
        source_count: Which panel this line belongs to.
        label: Which series this line is.

    Returns:
        `(exponents, residuals)`, both ascending in exponent.
    """
    selected = [
        (exponent, residual)
        for count, exponent, combinator, pooling, residual in zip(
            source_counts,
            phase_transform_exponents,
            pairwise_combinators,
            poolings,
            normalized_residuals,
            strict=True,
        )
        if count == source_count and build_series_label(combinator, pooling) == label
    ]
    if not selected:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    selected.sort()
    exponents = np.array([entry[0] for entry in selected], dtype=np.float64)
    residuals = np.array([entry[1] for entry in selected], dtype=np.float64)
    return exponents, residuals
