"""Panel 4 — the estimate against ground truth over the observations."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from app.export_controls import render_figure_with_export
from app.run_loader import load_run, read_ground_truth_series
from src.utils.visualization.receiver_signal_plotter import (
    plot_estimates_against_truth,
    plot_front_position_over_observations,
)

METRICS_ROOT: Path = Path("results/metrics")


def available_metrics_paths() -> list[Path]:
    """Localization metrics documents on disk, newest first.

    Returns:
        Paths, empty when the estimator has never been run.
    """
    if not METRICS_ROOT.is_dir():
        return []
    return sorted(METRICS_ROOT.glob("*.json"), key=lambda path: path.name)


def read_metrics(metrics_path: Path) -> dict[str, Any]:
    """Read one metrics document.

    Args:
        metrics_path: Path to the document.

    Returns:
        The parsed document.
    """
    document: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return document


def render(run_id: str) -> None:
    """Draw the estimate panel.

    Args:
        run_id: Identifier of the run to show front travel from.
    """
    st.header("Estimate")
    st.write(
        "Ground truth comes from the run directory; estimates come from a metrics "
        "document the estimator wrote. The two are never computed here."
    )

    run = load_run(run_id)
    st.subheader("Front travel over the run (F9)")
    render_figure_with_export(
        plot_front_position_over_observations(
            run.observation_times_s,
            read_ground_truth_series(run, "front_radius_m"),
            None,
        ),
        "f9_front_position",
    )

    metrics_paths = available_metrics_paths()
    if not metrics_paths:
        st.info("no localization metrics found; run the estimator to fill this panel")
        return

    st.subheader("Estimates against ground truth (F6)")
    selected = st.selectbox(
        "metrics document", [path.name for path in metrics_paths], index=0
    )
    metrics = read_metrics(METRICS_ROOT / selected)
    receiver_positions_xy_m = np.asarray(
        metrics["receiver_positions_xy_m"], dtype=np.float64
    )
    scenario_index = st.slider(
        "scenario", 1, len(metrics["scenarios"]), 1, help="One source position each."
    )
    scenario = metrics["scenarios"][scenario_index - 1]
    estimates = scenario["clip_estimates"]
    render_figure_with_export(
        plot_estimates_against_truth(
            np.asarray(
                [estimate["position_xy_m"] for estimate in estimates], dtype=np.float64
            ),
            np.asarray(
                [estimate["ellipse_semi_major_m"] for estimate in estimates],
                dtype=np.float64,
            ),
            np.asarray(
                [estimate["ellipse_semi_minor_m"] for estimate in estimates],
                dtype=np.float64,
            ),
            np.asarray(scenario["true_position_xy_m"], dtype=np.float64),
            receiver_positions_xy_m,
            f"scenario {scenario_index}",
        ),
        f"f6_estimates_scenario_{scenario_index}",
    )
    st.caption(
        f"median error {scenario['median_error_m']:.3f} m, "
        f"maximum {scenario['maximum_error_m']:.3f} m over "
        f"{len(estimates)} clips"
    )
