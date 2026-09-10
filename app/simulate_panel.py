"""Panel 6 — build a fire from scratch and watch it burn.

The live counterpart of `scripts/run_fire_simulation.py`: the same engines and
the same renderer, driven by widgets instead of a configuration file. Nothing
here decides anything scientific. Every parameter is assembled into a
`ForwardSimulationConfiguration` and handed to `build_simulation_context`, so
the composition root stays the only place a component is constructed.

Animation runs inside one script execution: the loop steps the fire, repaints
the placeholder and sleeps. Pressing Pause interrupts that script, which is how
Streamlit stops a long-running loop, so the button stays responsive without
rerunning once per frame. The loop yields every `SCRIPT_RUN_BUDGET_S` and
reruns, which keeps the session healthy over a long burn, and it stops on its
own once the fire is out.

The timestep slider is capped at the CFL limit from `time_step_calculator.py`,
so the fastest setting is also the fastest physically honest one: past that
bound the front skips cells and the shape starts depending on the timestep
rather than on the physics.
"""

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import streamlit as st

from app.image_display import paint_rgb_image
from app.run_loader import ladder_scene_options, load_configuration
from src.config.component_registry import (
    CELLULAR_AUTOMATON_ENGINE,
    PATCHY_TREES_FIELD,
    RANDOM_TREES_FIELD,
    RATE_OF_SPREAD_ENGINE,
    STATIC_SOURCE_ENGINE,
    UNIFORM_SCALAR_FIELD,
)
from src.config.simulation_configuration import ForwardSimulationConfiguration
from src.config.simulation_context import SimulationContext
from src.config.simulation_context_factory import build_simulation_context
from src.spark.acoustic.burning_cell_source_model import compute_fire_front_mask
from src.spark.fire.cellular_automaton_spread_engine import (
    FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
)
from src.spark.fire.fire_state import FireState
from src.spark.fire.rate_of_spread_equations import compute_rate_of_spread_balbi_2009
from src.spark.fire.time_step_calculator import compute_maximum_stable_time_step_s
from src.spark.terrain.square_grid_mesh import SquareGridMesh
from src.utils.array_types import Float64Array
from src.utils.visualization.fire_state_plotter import render_fire_state_rgb_image

SIMULATION_STATE_KEY: str = "live_simulation"
FRAME_INTERVAL_S: float = 0.10
SCRIPT_RUN_BUDGET_S: float = 20.0
MAXIMUM_CELL_COUNT: int = 400_000
CELLULAR_AUTOMATON_BURN_DURATION_S: float = 10.0
MINIMUM_TIME_STEP_FRACTION: float = 0.05
BURNOUT_PATIENCE_FRAMES: int = 12

ENGINE_LABELS: dict[str, str] = {
    RATE_OF_SPREAD_ENGINE: "Balbi rate of spread — physical",
    CELLULAR_AUTOMATON_ENGINE: "Cellular automaton — probabilistic",
    STATIC_SOURCE_ENGINE: "Static source — never spreads",
}
FUEL_FIELD_LABELS: dict[str, str] = {
    UNIFORM_SCALAR_FIELD: "Uniform — the same fuel everywhere",
    RANDOM_TREES_FIELD: "Random trees — a Poisson forest",
    PATCHY_TREES_FIELD: "Patchy trees — dense stands and clearings",
}


@dataclass
class LiveSimulation:
    """One fire being watched, and everything needed to keep stepping it.

    Attributes:
        config: The configuration the widgets assembled.
        context: The live objects the composition root built from it.
        state: The current fire state.
        fuel_density_fraction: The fuel field sampled once at the cell centres.
        burn_duration_s: How long a cell stays alight, used only for shading.
        time_step_s: Simulated seconds advanced per frame.
        maximum_time_step_s: The CFL-limited timestep for this scene.
        frame_index: How many frames have been drawn.
        is_playing: Whether the animation loop should keep stepping.
        ignition_log: Positions ignited by hand, for the caption.
        quiet_frame_count: Consecutive frames with nothing burning and nothing
            newly ignited. Used to notice that the fire is out.
        ignited_cell_count: Cells ignited as of the last frame.
    """

    config: ForwardSimulationConfiguration
    context: SimulationContext
    state: FireState
    fuel_density_fraction: Float64Array
    burn_duration_s: float
    time_step_s: float
    maximum_time_step_s: float
    frame_index: int = 0
    is_playing: bool = False
    ignition_log: list[tuple[float, float]] = field(default_factory=list)
    quiet_frame_count: int = 0
    ignited_cell_count: int = 0


def build_configuration_from_widgets(
    base: ForwardSimulationConfiguration,
    parameters: dict[str, Any],
) -> ForwardSimulationConfiguration:
    """Fold the widget values into a copy of the base scene.

    Args:
        base: The scene the widgets started from.
        parameters: Widget values, keyed by configuration field name.

    Returns:
        The configuration to build a context from.
    """
    return dataclasses.replace(
        base,
        mesh=dataclasses.replace(
            base.mesh,
            extent_x_m=parameters["extent_m"],
            extent_y_m=parameters["extent_m"],
            cell_spacing_m=parameters["cell_spacing_m"],
            use_diagonal_neighbors=parameters["use_diagonal_neighbors"],
        ),
        wind=dataclasses.replace(
            base.wind,
            wind_speed_m_per_s=parameters["wind_speed_m_per_s"],
            wind_bearing_rad=float(np.radians(parameters["wind_bearing_deg"])),
        ),
        fire=dataclasses.replace(
            base.fire,
            spread_engine_name=parameters["spread_engine_name"],
            fuel_field_type=parameters["fuel_field_type"],
            fuel_moisture_content_fraction=parameters["moisture_fraction"],
            tree_density_per_m2=parameters["tree_density_per_m2"],
            ignition_points_xy_fraction=(parameters["ignition_xy_fraction"],),
        ),
    )


def compute_maximum_time_step_s(
    config: ForwardSimulationConfiguration, context: SimulationContext
) -> float:
    """Largest timestep that still resolves the front, cell by cell.

    The same CFL bound the batch scripts use, from
    `src/spark/fire/time_step_calculator.py`: fire crossing the shortest mesh
    edge, scaled by the safety factor and clamped by the residence time so a
    cell cannot ignite and burn out between two frames.

    Args:
        config: The configuration the widgets assembled.
        context: The built context, carrying the mesh and the fuel bed.

    Returns:
        Simulated seconds per frame, at the CFL limit.
    """
    maximum_rate_of_spread_m_per_s = float(
        compute_rate_of_spread_balbi_2009(
            np.array([config.wind.wind_speed_m_per_s]), np.zeros(1), context.fuel
        )[0]
    )
    return compute_maximum_stable_time_step_s(
        context.mesh,
        maximum_rate_of_spread_m_per_s,
        config.fire.time_step_safety_factor,
        context.fuel.residence_time_s,
    )


@st.cache_resource(show_spinner="Building the scene…")
def build_context_for(
    config: ForwardSimulationConfiguration,
) -> tuple[SimulationContext, float]:
    """Build one scene and its CFL timestep, memoised on the configuration.

    The timestep slider needs the mesh before anything is drawn, so the context
    is built while the widgets are still being read. Caching keeps that off the
    critical path once a parameter set has been seen.

    Args:
        config: The configuration the widgets assembled.

    Returns:
        The context, and the largest stable timestep for it.
    """
    context = build_simulation_context(config)
    return context, compute_maximum_time_step_s(config, context)


def start_simulation(
    config: ForwardSimulationConfiguration,
    context: SimulationContext,
    time_step_s: float,
    maximum_time_step_s: float,
    is_playing: bool,
) -> LiveSimulation:
    """Ignite the configured points and hand back a simulation ready to run.

    The engine is stateful, so it is re-initialized here rather than reused
    from the cache: two simulations built from the same parameters must not
    share a bound domain.

    Args:
        config: The configuration the widgets assembled.
        context: The built context.
        time_step_s: Simulated seconds advanced per frame.
        maximum_time_step_s: The CFL limit, kept for the caption.
        is_playing: Whether to start animating immediately.

    Returns:
        The simulation, at time zero with its ignition points alight.
    """
    state = context.spread_engine.initialize(
        context.mesh, context.fuel_field, context.wind_field
    )
    state = context.spread_engine.ignite_cells(state, context.ignition_cell_indices)
    burn_duration_s = (
        context.fuel.residence_time_s
        if config.fire.spread_engine_name == RATE_OF_SPREAD_ENGINE
        else CELLULAR_AUTOMATON_BURN_DURATION_S
    )
    return LiveSimulation(
        config=config,
        context=context,
        state=state,
        fuel_density_fraction=np.clip(
            context.fuel_field.sample(context.mesh.cell_positions_xyz), 0.0, 1.0
        ),
        burn_duration_s=burn_duration_s,
        time_step_s=time_step_s,
        maximum_time_step_s=maximum_time_step_s,
        is_playing=is_playing,
        ignited_cell_count=int(np.count_nonzero(state.has_ignited)),
    )


def advance_one_frame(simulation: LiveSimulation) -> None:
    """Step the fire once and notice whether anything is still happening.

    Args:
        simulation: The simulation to advance, mutated in place.
    """
    simulation.state = simulation.context.spread_engine.step(
        simulation.state, simulation.time_step_s
    )
    simulation.frame_index += 1
    ignited_cell_count = int(np.count_nonzero(simulation.state.has_ignited))
    is_quiet = (
        not simulation.state.is_burning.any()
        and ignited_cell_count == simulation.ignited_cell_count
    )
    simulation.quiet_frame_count = simulation.quiet_frame_count + 1 if is_quiet else 0
    simulation.ignited_cell_count = ignited_cell_count


def is_fire_out(simulation: LiveSimulation) -> bool:
    """Report whether nothing can happen again without a fresh ignition.

    Args:
        simulation: The simulation to test.

    Returns:
        True once nothing has burnt or ignited for long enough that a pending
        arrival cannot still be in flight.
    """
    return simulation.quiet_frame_count >= BURNOUT_PATIENCE_FRAMES


def ignite_at_position(
    simulation: LiveSimulation, position_x_m: float, position_y_m: float
) -> None:
    """Light the cell nearest a position, in place.

    Args:
        simulation: The simulation to ignite in.
        position_x_m: Position along x, in metres.
        position_y_m: Position along y, in metres.
    """
    squared_distances_m2 = np.sum(
        (
            simulation.context.mesh.cell_positions_xyz[:, :2]
            - np.array([position_x_m, position_y_m], dtype=np.float64)
        )
        ** 2,
        axis=1,
    )
    cell_index = int(np.argmin(squared_distances_m2))
    simulation.state = simulation.context.spread_engine.ignite_cells(
        simulation.state, np.array([cell_index], dtype=np.int64)
    )
    simulation.ignition_log.append((position_x_m, position_y_m))


def render_frame(simulation: LiveSimulation) -> npt.NDArray[np.uint8]:
    """Paint the current fire state as an RGB image.

    Args:
        simulation: The simulation to draw.

    Returns:
        Uint8 array of shape (n_y, n_x, 3).

    Raises:
        TypeError: If the mesh is not a square grid, which a raster needs.
    """
    mesh = simulation.context.mesh
    if not isinstance(mesh, SquareGridMesh):
        raise TypeError("the live display needs a square grid mesh")
    return render_fire_state_rgb_image(
        simulation.state,
        mesh,
        simulation.fuel_density_fraction,
        simulation.burn_duration_s,
        FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
    )


def describe_frame(simulation: LiveSimulation) -> str:
    """One line of counts for the caption under the image.

    Args:
        simulation: The simulation to describe.

    A frame with nothing alight says so. The burning band is one cell thick,
    so it can empty for a frame or two while an arrival is still in flight, and
    a still picture with no note reads as a frozen application.

    Returns:
        A caption naming the time and the cell counts.
    """
    is_on_front = compute_fire_front_mask(
        simulation.state, simulation.context.mesh.neighbor_indices
    )
    burnt_area_m2 = (
        int(np.count_nonzero(simulation.state.has_ignited))
        * simulation.config.mesh.cell_spacing_m**2
    )
    burning_cell_count = int(np.count_nonzero(simulation.state.is_burning))
    quiet_note = (
        "  ·  quiet, nothing alight this frame" if burning_cell_count == 0 else ""
    )
    return (
        f"t = {simulation.state.current_time_s:.0f} s  ·  "
        f"burning {burning_cell_count}  ·  "
        f"front {int(np.count_nonzero(is_on_front))}  ·  "
        f"ignited {int(np.count_nonzero(simulation.state.has_ignited))}  ·  "
        f"burnt {burnt_area_m2:.0f} m²  ·  "
        f"frame {simulation.frame_index}"
        f"{quiet_note}"
    )


def render_parameter_controls(
    base: ForwardSimulationConfiguration,
) -> dict[str, Any]:
    """Draw every parameter widget and read it back.

    Args:
        base: The scene the widgets start from.

    Returns:
        The parameter values, keyed by configuration field name.
    """
    grid_column, wind_column, fuel_column = st.columns(3)
    with grid_column:
        st.markdown("**Grid**")
        extent_m = st.slider("extent (m)", 20.0, 200.0, base.mesh.extent_x_m, 10.0)
        cell_spacing_m = st.select_slider(
            "cell spacing (m)", [0.25, 0.5, 1.0, 2.0], value=base.mesh.cell_spacing_m
        )
        use_diagonal_neighbors = st.checkbox(
            "8-connectivity", value=base.mesh.use_diagonal_neighbors
        )
    with wind_column:
        st.markdown("**Wind**")
        wind_speed_m_per_s = st.slider(
            "speed (m/s)", 0.0, 12.0, base.wind.wind_speed_m_per_s, 0.5
        )
        wind_bearing_deg = st.slider(
            "bearing (° from +x)", 0, 359, int(np.degrees(base.wind.wind_bearing_rad))
        )
    with fuel_column:
        st.markdown("**Fuel**")
        fuel_field_type = st.selectbox(
            "field",
            list(FUEL_FIELD_LABELS),
            index=list(FUEL_FIELD_LABELS).index(base.fire.fuel_field_type),
            format_func=lambda name: FUEL_FIELD_LABELS[name],
        )
        moisture_fraction = (
            st.slider(
                "moisture (%)",
                2.0,
                40.0,
                100.0 * base.fire.fuel_moisture_content_fraction,
                1.0,
            )
            / 100.0
        )
        tree_density_per_m2 = st.slider(
            "tree density (per m²)",
            0.01,
            0.40,
            base.fire.tree_density_per_m2,
            0.01,
            disabled=fuel_field_type == UNIFORM_SCALAR_FIELD,
        )

    engine_column, ignition_column = st.columns(2)
    with engine_column:
        st.markdown("**Engine**")
        spread_engine_name = st.selectbox(
            "model",
            list(ENGINE_LABELS),
            index=list(ENGINE_LABELS).index(base.fire.spread_engine_name),
            format_func=lambda name: ENGINE_LABELS[name],
        )
    with ignition_column:
        st.markdown("**First ignition**")
        ignition_x_fraction = st.slider(
            "x (fraction of extent)", 0.0, 1.0, base.fire.ignition_cell_x_fraction, 0.01
        )
        ignition_y_fraction = st.slider(
            "y (fraction of extent)", 0.0, 1.0, base.fire.ignition_cell_y_fraction, 0.01
        )

    return {
        "extent_m": extent_m,
        "cell_spacing_m": cell_spacing_m,
        "use_diagonal_neighbors": use_diagonal_neighbors,
        "wind_speed_m_per_s": wind_speed_m_per_s,
        "wind_bearing_deg": wind_bearing_deg,
        "fuel_field_type": fuel_field_type,
        "moisture_fraction": moisture_fraction,
        "tree_density_per_m2": tree_density_per_m2,
        "spread_engine_name": spread_engine_name,
        "ignition_xy_fraction": (ignition_x_fraction, ignition_y_fraction),
    }


def render_time_controls(maximum_time_step_s: float) -> tuple[float, bool]:
    """Draw the timestep slider, bounded above by the CFL limit.

    The slider tops out at the largest timestep that still resolves the front
    cell by cell, so the fastest setting is also the fastest physically honest
    one. Going past it would let the front skip cells, and the shape would
    start depending on the timestep instead of on the physics.

    Args:
        maximum_time_step_s: The CFL limit for the current parameters.

    Returns:
        The chosen timestep in seconds, and whether to start playing on build.
    """
    st.markdown("**Time**")
    slider_column, autoplay_column = st.columns([3, 1])
    with slider_column:
        time_step_s = st.slider(
            "simulated seconds per frame",
            min_value=round(MINIMUM_TIME_STEP_FRACTION * maximum_time_step_s, 3),
            max_value=round(maximum_time_step_s, 3),
            value=round(maximum_time_step_s, 3),
            step=round(MINIMUM_TIME_STEP_FRACTION * maximum_time_step_s, 3),
            help=(
                "Capped at the CFL limit from time_step_calculator.py: fire "
                "crossing the shortest mesh edge, scaled by the safety factor "
                "and clamped by the fuel residence time."
            ),
        )
    with autoplay_column:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        should_autoplay = st.checkbox("play on build", value=True)
    st.caption(
        f"CFL limit {maximum_time_step_s:.2f} s per frame  ·  "
        f"{time_step_s / FRAME_INTERVAL_S:.0f} simulated seconds per real second "
        f"at this setting"
    )
    return float(time_step_s), should_autoplay


def choose_default_scene_label(scene_options: dict[str, Path]) -> str:
    """Pick which scene the picker starts on.

    The first rung of the ladder uses the static source engine, which never
    spreads: starting there means pressing Play and watching nothing happen.
    A spreading scene is picked instead whenever one exists.

    Args:
        scene_options: Picker labels mapped to configuration directories.

    Returns:
        The label to select by default.
    """
    for label, directory in scene_options.items():
        if (
            load_configuration(str(directory)).fire.spread_engine_name
            != STATIC_SOURCE_ENGINE
        ):
            return label
    return next(iter(scene_options))


def render() -> None:
    """Draw the live simulation panel."""
    st.header("Simulate")
    st.write(
        "Build a fire from scratch and watch it burn. Same engines and same "
        "renderer as the batch scripts, driven by these widgets instead of a "
        "configuration file."
    )

    scene_options = ladder_scene_options()
    if not scene_options:
        st.warning("no ladder configuration directories found")
        return

    scene_labels = list(scene_options)
    base_label = st.selectbox(
        "start from",
        scene_labels,
        index=scene_labels.index(choose_default_scene_label(scene_options)),
        key="simulate_base_scene",
    )
    base = load_configuration(str(scene_options[base_label]))

    parameters = render_parameter_controls(base)
    config = build_configuration_from_widgets(base, parameters)
    cell_count = (round(parameters["extent_m"] / parameters["cell_spacing_m"]) + 1) ** 2
    if cell_count > MAXIMUM_CELL_COUNT:
        st.error(
            f"{cell_count:,} cells is too many to animate; widen the cell "
            f"spacing or shrink the extent"
        )
        return

    context, maximum_time_step_s = build_context_for(config)
    time_step_s, should_autoplay = render_time_controls(maximum_time_step_s)
    st.caption(f"{cell_count:,} cells")

    simulation: LiveSimulation | None = st.session_state.get(SIMULATION_STATE_KEY)
    build_column, play_column, step_column, reset_column = st.columns(4)
    with build_column:
        if st.button("Build and run", type="primary", use_container_width=True):
            st.session_state[SIMULATION_STATE_KEY] = start_simulation(
                config, context, time_step_s, maximum_time_step_s, should_autoplay
            )
            st.rerun()
    with play_column:
        play_label = (
            "Pause" if simulation is not None and simulation.is_playing else "Play"
        )
        if (
            st.button(play_label, use_container_width=True, disabled=simulation is None)
            and simulation is not None
        ):
            simulation.is_playing = not simulation.is_playing
            if simulation.is_playing:
                simulation.quiet_frame_count = 0
            st.rerun()
    with step_column:
        if (
            st.button("Step", use_container_width=True, disabled=simulation is None)
            and simulation is not None
        ):
            advance_one_frame(simulation)
    with reset_column:
        if st.button("Reset", use_container_width=True, disabled=simulation is None):
            st.session_state.pop(SIMULATION_STATE_KEY, None)
            st.rerun()

    if simulation is None:
        st.info("Set the parameters above, then press Build and run.")
        return

    if simulation.time_step_s != time_step_s:
        simulation.time_step_s = time_step_s

    st.caption(
        f"dt = {simulation.time_step_s:.2f} s per frame of a "
        f"{simulation.maximum_time_step_s:.2f} s CFL limit  ·  "
        f"{ENGINE_LABELS[simulation.config.fire.spread_engine_name]}"
    )

    image_placeholder = st.empty()
    caption_placeholder = st.empty()
    notice_placeholder = st.empty()

    with st.expander("Light another fire"):
        extra_x_column, extra_y_column, ignite_column = st.columns([2, 2, 1])
        with extra_x_column:
            extra_x_m = st.slider(
                "x (m)", 0.0, simulation.config.mesh.extent_x_m, 0.0, 0.5
            )
        with extra_y_column:
            extra_y_m = st.slider(
                "y (m)", 0.0, simulation.config.mesh.extent_y_m, 0.0, 0.5
            )
        with ignite_column:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("Ignite here", use_container_width=True):
                ignite_at_position(simulation, extra_x_m, extra_y_m)
                simulation.quiet_frame_count = 0
        if simulation.ignition_log:
            st.caption(
                "lit by hand: "
                + ", ".join(f"({x:.0f}, {y:.0f}) m" for x, y in simulation.ignition_log)
            )

    paint_rgb_image(image_placeholder, render_frame(simulation))
    caption_placeholder.caption(describe_frame(simulation))

    if simulation.config.fire.spread_engine_name == STATIC_SOURCE_ENGINE:
        notice_placeholder.info(
            "The static source engine never spreads, so the picture will not "
            "change. Pick another engine to watch a front move."
        )
        return

    if not simulation.is_playing:
        return

    deadline = time.monotonic() + SCRIPT_RUN_BUDGET_S
    while time.monotonic() < deadline:
        advance_one_frame(simulation)
        paint_rgb_image(image_placeholder, render_frame(simulation))
        caption_placeholder.caption(describe_frame(simulation))
        if is_fire_out(simulation):
            simulation.is_playing = False
            notice_placeholder.success(
                f"The fire is out at t = {simulation.state.current_time_s:.0f} s. "
                "Light another fire, or press Reset."
            )
            return
        time.sleep(FRAME_INTERVAL_S)
    st.rerun()
