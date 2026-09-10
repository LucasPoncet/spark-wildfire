"""Sweep curves: how localization degrades as one setting is varied."""

import numpy as np
from matplotlib.figure import Figure

from src.utils.array_types import Float64Array


def plot_metric_against_parameter(
    parameter_values: Float64Array,
    metric_values: Float64Array,
    series_labels: list[str],
    parameter_label: str,
    metric_label: str,
    title: str,
) -> Figure:
    """Draws one metric against one swept parameter, one line per series.

    Args:
        parameter_values: Swept values, shape `(n_values,)`.
        metric_values: Metric per series and value, shape
            `(n_series, n_values)`.
        series_labels: One label per series.
        parameter_label: Axis label for the swept parameter.
        metric_label: Axis label for the metric.
        title: Figure title.

    Returns:
        The figure. Saving it is the caller's job.

    Raises:
        ValueError: If the label count does not match the series count.
    """
    values = np.asarray(parameter_values, dtype=np.float64)
    metrics = np.atleast_2d(np.asarray(metric_values, dtype=np.float64))
    if metrics.shape[0] != len(series_labels):
        raise ValueError("one label is required per metric series")

    figure = Figure(figsize=(7.0, 4.5))
    axes = figure.add_subplot(111)
    for series, label in zip(metrics, series_labels, strict=True):
        axes.plot(values, series, marker="o", label=label)
    axes.set_xlabel(parameter_label)
    axes.set_ylabel(metric_label)
    axes.set_title(title)
    axes.grid(visible=True, alpha=0.3)
    if len(series_labels) > 1:
        axes.legend()
    figure.tight_layout()
    return figure


def plot_source_count_confusion(
    true_source_counts: Float64Array,
    estimated_source_counts: Float64Array,
    title: str,
) -> Figure:
    """Draws estimated source count against true source count.

    A point off the diagonal is the failure the sweep exists to find: a merged
    pair below it, a spurious detection above it.

    Args:
        true_source_counts: True counts, shape `(n_runs,)`.
        estimated_source_counts: Estimated counts, same shape.
        title: Figure title.

    Returns:
        The figure. Saving it is the caller's job.
    """
    true_counts = np.asarray(true_source_counts, dtype=np.float64)
    estimated_counts = np.asarray(estimated_source_counts, dtype=np.float64)
    figure = Figure(figsize=(5.0, 5.0))
    axes = figure.add_subplot(111)
    axes.scatter(true_counts, estimated_counts, s=60, alpha=0.7, color="tab:blue")
    upper_limit = float(
        max(true_counts.max(initial=1.0), estimated_counts.max(initial=1.0)) + 1.0
    )
    limits = (0.0, upper_limit)
    axes.plot(limits, limits, linestyle="--", color="grey", linewidth=1.0)
    axes.set_xlabel("true source count")
    axes.set_ylabel("estimated source count")
    axes.set_title(title)
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    return figure
