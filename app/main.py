"""SPARK — the defense demo and the figure exporter, in one application.

Run it with:

    uv run --group app streamlit run app/main.py

Holds no plotting logic and no physics. It is a run picker, a set of parameter
widgets, calls into `simulation_context_factory` and the plotters, and export
buttons. Anything that cannot be expressed that way belongs in `src/`, and
nothing in `src/` may import this package.

The panels are ordered the way the defense should walk through them: what the
scene is, what the fire does, a fire you build and run yourself, what the
channel does to it, what the estimator recovers, and where all of that stops
working. The last panel launches a full acoustic render in the background.
"""

import streamlit as st

from app import (
    channel_panel,
    estimate_panel,
    failure_modes_panel,
    physics_panel,
    scene_panel,
    simulate_panel,
)
from app.live_mode import render_live_mode
from app.run_loader import available_run_ids

PANEL_NAMES: tuple[str, ...] = (
    "Scene",
    "Physics",
    "Simulate",
    "Channel and receivers",
    "Estimate",
    "Failure modes",
    "Render a run",
)


def main() -> None:
    """Draw the application."""
    st.set_page_config(page_title="SPARK", layout="wide")
    st.title("SPARK — acoustic wildfire kinematics")

    run_ids = available_run_ids()
    with st.sidebar:
        st.header("Run")
        if run_ids:
            selected_run_id = st.selectbox("replay", run_ids, index=0)
            st.caption(f"{len(run_ids)} runs on disk, newest first")
        else:
            selected_run_id = ""
            st.warning(
                "no runs found. Render one with "
                "`uv run python scripts/run_acoustic_rendering.py`."
            )
        panel_name = st.radio("panel", PANEL_NAMES, index=0)

    if panel_name == "Scene":
        scene_panel.render()
    elif panel_name == "Physics":
        physics_panel.render()
    elif panel_name == "Simulate":
        simulate_panel.render()
    elif panel_name == "Render a run":
        render_live_mode()
    elif not selected_run_id:
        st.info("this panel needs a saved run")
    elif panel_name == "Channel and receivers":
        channel_panel.render(selected_run_id)
    elif panel_name == "Estimate":
        estimate_panel.render(selected_run_id)
    else:
        failure_modes_panel.render()


if __name__ == "__main__":
    main()
