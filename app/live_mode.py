"""Launching a render from the application, guarded.

A full render is minutes, so it runs in a background thread with a progress
readout and a cancel, and the demo defaults to replay. Live mode is for
exploration, not for standing in front of a supervisor.

The thread runs `scripts/run_acoustic_rendering.py` as a subprocess rather than
importing it, so a cancelled run leaves nothing half-built inside this process.
"""

import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from app.run_loader import ladder_scene_options

POLL_INTERVAL_S: float = 1.0
LIVE_RUN_STATE_KEY: str = "live_run"


@dataclass
class LiveRun:
    """One background render and what it has printed so far.

    Attributes:
        configuration_directory: Scene the render was launched from.
        process: The running subprocess, or None once it has finished.
        output_lines: Everything the render has printed.
        is_finished: Whether the render has stopped, cancelled or not.
        return_code: Exit status once finished.
    """

    configuration_directory: str
    process: subprocess.Popen[str] | None = None
    output_lines: list[str] = field(default_factory=list)
    is_finished: bool = False
    return_code: int | None = None


def start_render(live_run: LiveRun) -> None:
    """Launch the rendering script and stream its output into the run.

    Args:
        live_run: The record to fill in as the render proceeds.
    """
    live_run.process = subprocess.Popen(
        [
            sys.executable,
            str(Path("scripts") / "run_acoustic_rendering.py"),
            "--configs",
            live_run.configuration_directory,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def drain_output() -> None:
        if live_run.process is None or live_run.process.stdout is None:
            return
        for line in live_run.process.stdout:
            live_run.output_lines.append(line.rstrip())
        live_run.return_code = live_run.process.wait()
        live_run.is_finished = True

    threading.Thread(target=drain_output, daemon=True).start()


def render_live_mode() -> None:
    """Draw the live-run panel."""
    st.header("Live run")
    st.warning(
        "A full render takes minutes. The demo should replay a saved run; use "
        "this to explore a new scene, not to fill a silence in front of a room."
    )

    scene_options = ladder_scene_options()
    if not scene_options:
        st.info("no ladder configuration directories found")
        return

    live_run: LiveRun | None = st.session_state.get(LIVE_RUN_STATE_KEY)

    if live_run is None or live_run.is_finished:
        selected_label = st.selectbox(
            "scene to render", list(scene_options), index=0, key="live_scene"
        )
        if st.button("Start render", type="primary"):
            started = LiveRun(
                configuration_directory=str(scene_options[selected_label])
            )
            start_render(started)
            st.session_state[LIVE_RUN_STATE_KEY] = started
            st.rerun()

    if live_run is None:
        return

    st.subheader(f"Rendering {live_run.configuration_directory}")
    st.code("\n".join(live_run.output_lines[-20:]) or "starting…", language="text")

    if live_run.is_finished:
        if live_run.return_code == 0:
            st.success("render finished; pick it in the sidebar to replay it")
        else:
            st.error(f"render exited with status {live_run.return_code}")
        if st.button("Clear"):
            del st.session_state[LIVE_RUN_STATE_KEY]
            st.rerun()
        return

    if st.button("Cancel"):
        if live_run.process is not None:
            live_run.process.terminate()
        live_run.is_finished = True
        st.rerun()

    st.caption("still running…")
    st.button("Refresh", key="live_refresh")
