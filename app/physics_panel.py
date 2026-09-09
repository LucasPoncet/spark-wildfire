"""Panel 2 — the fire model and the front it produces."""

import numpy as np
import streamlit as st

from app.export_controls import render_figure_with_export
from app.run_loader import (
    ladder_scene_options,
    load_configuration,
    load_context,
)
from src.config.simulation_context import SimulationContext
from src.spark.acoustic.burning_cell_source_model import compute_fire_front_mask
from src.spark.fire.fire_state import FireState
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.terrain.square_grid_mesh import SquareGridMesh
from src.utils.array_types import BoolArray
from src.utils.visualization.fire_state_plotter import plot_ignition_time_map
from src.utils.visualization.rate_of_spread_plotter import plot_rate_of_spread_panels

WIND_SPEED_SWEEP_M_PER_S = np.linspace(0.0, 12.0, 120)
MOISTURE_SWEEP_FRACTION = np.linspace(0.02, 0.30, 60)
FRONT_SNAPSHOT_COUNT: int = 4


def replay_fire(
    context: SimulationContext, duration_s: float, time_step_s: float
) -> tuple[FireState, list[BoolArray], list[float]]:
    """Run the fire alone, keeping a few front masks along the way.

    The fire without acoustics costs well under a second, which is why the run
    directory stores aggregates rather than per-cell state.

    Args:
        context: The scene's live objects.
        duration_s: Simulated seconds to run for.
        time_step_s: The fire timestep, in seconds.

    Returns:
        The final state, the front masks and their simulation times.
    """
    state = context.spread_engine.initialize(
        context.mesh, context.fuel_field, context.wind_field
    )
    state = context.spread_engine.ignite_cells(state, context.ignition_cell_indices)
    step_count = max(1, int(duration_s / time_step_s))
    snapshot_steps = {
        round(fraction * step_count)
        for fraction in np.linspace(0.25, 1.0, FRONT_SNAPSHOT_COUNT)
    }
    front_masks: list[BoolArray] = []
    front_mask_times_s: list[float] = []
    for step_index in range(1, step_count + 1):
        state = context.spread_engine.step(state, time_step_s)
        if step_index in snapshot_steps:
            front_masks.append(
                compute_fire_front_mask(state, context.mesh.neighbor_indices)
            )
            front_mask_times_s.append(float(state.current_time_s))
    return state, front_masks, front_mask_times_s


def render() -> None:
    """Draw the physics panel."""
    st.header("Physics")
    st.write(
        "Rate of spread against wind, slope and moisture, and the front the model "
        "produces. The heatmap is the whole run at once: contours of equal "
        "ignition time are the front's successive positions."
    )

    slope_degrees = st.multiselect(
        "slopes to draw (degrees)", [0, 10, 20, 30, 40], default=[0, 10, 20, 30]
    )
    render_figure_with_export(
        plot_rate_of_spread_panels(
            WIND_SPEED_SWEEP_M_PER_S,
            np.radians(np.array(sorted(slope_degrees) or [0], dtype=np.float64)),
            MOISTURE_SWEEP_FRACTION,
            FuelProperties.pine_needle_litter(),
        ),
        "f1_rate_of_spread",
    )

    scene_options = ladder_scene_options()
    if not scene_options:
        st.warning("no ladder configuration directories found")
        return

    selected_label = st.selectbox("scene", list(scene_options), index=0, key="physics")
    configuration_directory = scene_options[selected_label]
    config = load_configuration(str(configuration_directory))
    context = load_context(str(configuration_directory))
    if not isinstance(context.mesh, SquareGridMesh):
        st.warning("the ignition-time raster needs a square grid mesh")
        return

    maximum_rate_of_spread_m_per_s = float(
        compute_rate_of_spread_balbi_2009(
            np.array([config.wind.wind_speed_m_per_s]), np.zeros(1), context.fuel
        )[0]
    )
    time_step_s = compute_maximum_stable_time_step_s(
        context.mesh,
        maximum_rate_of_spread_m_per_s,
        config.fire.time_step_safety_factor,
        context.fuel.residence_time_s,
    )
    st.caption(
        f"dt = {time_step_s:.1f} s, "
        f"R = {maximum_rate_of_spread_m_per_s * 1000.0:.2f} mm/s at "
        f"{config.wind.wind_speed_m_per_s:.1f} m/s wind"
    )

    final_state, front_masks, front_mask_times_s = replay_fire(
        context, config.fire.simulation_duration_s, time_step_s
    )
    render_figure_with_export(
        plot_ignition_time_map(
            final_state, context.mesh, front_masks, front_mask_times_s
        ),
        "f3_front_evolution",
    )
