"""Panel 3 — the channel, and what it does to the receivers."""

import numpy as np
import streamlit as st

from app.export_controls import render_figure_with_export
from app.run_loader import compute_receiver_levels_db, load_run
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.utils.visualization.channel_plotter import plot_channel_panels
from src.utils.visualization.receiver_signal_plotter import plot_receiver_level_traces

ABSORPTION_FREQUENCY_SWEEP_HZ = np.geomspace(50.0, 20000.0, 300)
OCTAVE_BAND_CENTRES_HZ = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
CHANNEL_RANGE_SWEEP_M = np.linspace(1.0, 150.0, 150)


def render(run_id: str) -> None:
    """Draw the channel panel.

    Args:
        run_id: Identifier of the run to show levels from.
    """
    st.header("Channel and receivers")
    st.write(
        "Absorption rises with frequency, so the high bands lose more over the "
        "same path. That growing spread between bands is the range information "
        "an estimator inverts."
    )

    run = load_run(run_id)
    atmosphere = run.config["atmosphere"]
    columns = st.columns(3)
    with columns[0]:
        air_temperature_celsius = st.slider(
            "air temperature (°C)",
            -10.0,
            45.0,
            float(atmosphere["air_temperature_celsius"]),
            1.0,
        )
    with columns[1]:
        relative_humidity_percent = st.slider(
            "relative humidity (%)",
            5.0,
            100.0,
            float(atmosphere["relative_humidity_percent"]),
            5.0,
        )
    with columns[2]:
        reference_distance_m = st.number_input(
            "reference distance (m)",
            min_value=0.1,
            max_value=10.0,
            value=float(run.config["acoustic_rendering"]["reference_distance_m"]),
            step=0.1,
        )

    conditions = AtmosphericConditions(
        air_temperature_celsius=air_temperature_celsius,
        relative_humidity_percent=relative_humidity_percent,
        pressure_kpa=float(atmosphere["pressure_kpa"]),
    )
    render_figure_with_export(
        plot_channel_panels(
            ABSORPTION_FREQUENCY_SWEEP_HZ,
            OCTAVE_BAND_CENTRES_HZ,
            CHANNEL_RANGE_SWEEP_M,
            conditions,
            float(reference_distance_m),
        ),
        "f4_channel",
    )

    st.subheader("Received levels over the run")
    render_figure_with_export(
        plot_receiver_level_traces(
            run.observation_times_s,
            compute_receiver_levels_db(run),
            run.receiver_positions_xy_m,
        ),
        "f5_receiver_levels",
    )
