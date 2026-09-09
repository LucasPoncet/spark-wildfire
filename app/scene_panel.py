"""Panel 1 — the scene, and where it cannot be inverted."""

import streamlit as st

from app.export_controls import render_figure_with_export
from app.run_loader import (
    ladder_scene_options,
    load_configuration,
    load_context,
)
from src.config.simulation_configuration import (
    DEFAULT_CONFIGURATION_DIRECTORY,
    load_localization_configuration,
)
from src.utils.visualization.mesh_plotter import (
    plot_scene_geometry,
    plot_scene_geometry_panels,
)

LOCALIZATION_FILENAME: str = "localization.toml"


def read_near_singular_tolerance() -> float:
    """Read the estimator's degeneracy tolerance from the shipped configuration.

    Returns:
        The tolerance, as a distance of the level ratio from unity.
    """
    return load_localization_configuration(
        DEFAULT_CONFIGURATION_DIRECTORY / LOCALIZATION_FILENAME
    ).triangulation.near_singular_tolerance


def render() -> None:
    """Draw the scene panel."""
    st.header("Scene")
    st.write(
        "Every rung of the ladder is a configuration directory. The shaded band "
        "is where the level ratio is within tolerance of one: a source inside it "
        "is unrecoverable however clean the signal."
    )

    scene_options = ladder_scene_options()
    if not scene_options:
        st.warning("no ladder configuration directories found")
        return

    default_tolerance = read_near_singular_tolerance()
    tolerance = st.slider(
        "near-singular tolerance",
        min_value=0.01,
        max_value=0.40,
        value=float(default_tolerance),
        step=0.01,
        help="Smallest trusted distance of the level ratio from unity.",
    )

    selected_label = st.selectbox("scene", list(scene_options), index=0)
    configuration_directory = scene_options[selected_label]
    config = load_configuration(str(configuration_directory))
    context = load_context(str(configuration_directory))
    st.caption(config.experiment.description)

    render_figure_with_export(
        plot_scene_geometry(
            config.mesh.extent_x_m,
            config.mesh.extent_y_m,
            context.mesh.cell_positions_xyz[context.ignition_cell_indices, :2],
            context.receiver_positions_xy_m,
            tolerance,
            config.experiment.name,
        ),
        f"scene_{config.experiment.name}",
    )

    with st.expander("All rungs side by side (F2)"):
        titles: list[str] = []
        extents_xy_m: list[tuple[float, float]] = []
        ignition_positions_xy_m = []
        receiver_positions_xy_m = []
        for directory in scene_options.values():
            panel_config = load_configuration(str(directory))
            panel_context = load_context(str(directory))
            titles.append(panel_config.experiment.name)
            extents_xy_m.append(
                (panel_config.mesh.extent_x_m, panel_config.mesh.extent_y_m)
            )
            ignition_positions_xy_m.append(
                panel_context.mesh.cell_positions_xyz[
                    panel_context.ignition_cell_indices, :2
                ]
            )
            receiver_positions_xy_m.append(panel_context.receiver_positions_xy_m)
        render_figure_with_export(
            plot_scene_geometry_panels(
                titles,
                extents_xy_m,
                ignition_positions_xy_m,
                receiver_positions_xy_m,
                tolerance,
            ),
            "f2_scene_geometry",
        )
