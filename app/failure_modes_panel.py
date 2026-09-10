"""Panel 5 — where the estimator fails, and where its uncertainty is wrong."""

from pathlib import Path

import numpy as np
import streamlit as st

from app.estimate_panel import available_metrics_paths, read_metrics
from app.export_controls import render_figure_with_export
from src.utils.array_types import Float64Array
from src.utils.visualization.mesh_plotter import compute_near_singular_band_mask
from src.utils.visualization.receiver_signal_plotter import (
    plot_error_against_bisector_distance,
    plot_reduced_chi_square_by_scenario,
)

METRICS_ROOT: Path = Path("results/metrics")
NEAR_SINGULAR_BAND_FRACTION_OF_BASELINE: float = 0.05


def compute_bisector_distances_m(
    true_positions_xy_m: Float64Array,
    receiver_positions_xy_m: Float64Array,
) -> Float64Array:
    """Distance of each source from the receiver pair's perpendicular bisector.

    Args:
        true_positions_xy_m: Float64 array of shape (n_scenarios, 2).
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).

    Returns:
        Float64 array of shape (n_scenarios,), in metres.
    """
    midpoint_xy_m = receiver_positions_xy_m[:2].mean(axis=0)
    baseline_xy_m = receiver_positions_xy_m[1] - receiver_positions_xy_m[0]
    baseline_direction = baseline_xy_m / np.linalg.norm(baseline_xy_m)
    return np.asarray(
        np.abs((true_positions_xy_m - midpoint_xy_m) @ baseline_direction),
        dtype=np.float64,
    )


def render() -> None:
    """Draw the failure modes panel."""
    st.header("Failure modes")
    st.write(
        "Two things break independently. Near the perpendicular bisector the "
        "geometry carries no information, so no estimator can work. Separately, "
        "the reported uncertainties are not yet calibrated: positions are "
        "accurate to centimetres while reduced chi-square runs from 18 to 78."
    )

    metrics_paths = available_metrics_paths()
    if not metrics_paths:
        st.info("no localization metrics found; run the estimator to fill this panel")
        return

    selected = st.selectbox(
        "metrics document",
        [path.name for path in metrics_paths],
        index=0,
        key="failure_modes",
    )
    metrics = read_metrics(METRICS_ROOT / selected)
    receiver_positions_xy_m = np.asarray(
        metrics["receiver_positions_xy_m"], dtype=np.float64
    )
    true_positions_xy_m = np.asarray(
        [scenario["true_position_xy_m"] for scenario in metrics["scenarios"]],
        dtype=np.float64,
    )
    errors_m = np.asarray(
        [scenario["median_error_m"] for scenario in metrics["scenarios"]],
        dtype=np.float64,
    )
    baseline_m = float(
        np.linalg.norm(receiver_positions_xy_m[1] - receiver_positions_xy_m[0])
    )

    st.subheader("Error against degeneracy (F7)")
    band_half_width_m = st.slider(
        "near-singular band half width (m)",
        0.0,
        baseline_m / 2.0,
        baseline_m * NEAR_SINGULAR_BAND_FRACTION_OF_BASELINE,
        0.5,
    )
    render_figure_with_export(
        plot_error_against_bisector_distance(
            compute_bisector_distances_m(true_positions_xy_m, receiver_positions_xy_m),
            errors_m,
            band_half_width_m,
        ),
        "f7_error_against_bisector_distance",
    )

    st.subheader("Move a source towards the bisector")
    columns = st.columns(2)
    with columns[0]:
        probe_x_m = st.slider(
            "source x (m)", 0.0, float(metrics["domain_size_xy_m"][0]), 25.0, 0.5
        )
    with columns[1]:
        probe_y_m = st.slider(
            "source y (m)", 0.0, float(metrics["domain_size_xy_m"][1]), 70.0, 0.5
        )
    tolerance = st.slider("near-singular tolerance", 0.01, 0.40, 0.05, 0.01)
    is_degenerate = bool(
        compute_near_singular_band_mask(
            np.array([[probe_x_m, probe_y_m]]),
            receiver_positions_xy_m[0],
            receiver_positions_xy_m[1],
            tolerance,
        )[0]
    )
    if is_degenerate:
        st.error("near-singular: this source is inside the degenerate band")
    else:
        st.success("outside the degenerate band")

    st.subheader("Uncertainty calibration (F8)")
    render_figure_with_export(
        plot_reduced_chi_square_by_scenario(
            [f"S{index + 1}" for index in range(len(metrics["scenarios"]))],
            [
                np.asarray(
                    [
                        estimate["reduced_chi_square"]
                        for estimate in scenario["clip_estimates"]
                    ],
                    dtype=np.float64,
                )
                for scenario in metrics["scenarios"]
            ],
            [
                all(
                    estimate["is_near_singular"]
                    for estimate in scenario["clip_estimates"]
                )
                for scenario in metrics["scenarios"]
            ],
        ),
        "f8_reduced_chi_square",
    )
    st.caption(
        "Reduced chi-square is the estimator's own consistency check. Each octave "
        "band gives an independent estimate of the same geometric level difference, "
        "with a variance the estimator predicts for it; reduced chi-square is the "
        "spread the bands actually show divided by the spread they were predicted "
        "to show. One means the bands disagree by exactly as much as the stated "
        "uncertainties allow, so the error ellipse can be believed. Twenty to "
        "eighty, as here, means they disagree far more than predicted — the "
        "positions are still right to centimetres, but the error bars on them are "
        "not yet trustworthy. A scenario marked degenerate has no fit at all: on "
        "the bisector there is nothing for the bands to agree about. This is the "
        "estimator owner's open problem; the figure reports it rather than hiding it."
    )
