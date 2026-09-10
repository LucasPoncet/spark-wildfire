"""One export control, shared by every panel.

A figure shown in the demo and a figure in the report come from the same
plotter call and the same writer, so they cannot differ. The button writes
into the directory `scripts/export_report_figures.py` writes into.
"""

import io
from pathlib import Path

import streamlit as st
from matplotlib.figure import Figure

FIGURE_ROOT: Path = Path("results/figures")


def render_figure_with_export(figure: Figure, figure_name: str) -> None:
    """Show one figure and offer to save it as the report's SVG.

    Args:
        figure: The figure a plotter returned.
        figure_name: Stable name, without extension, matching the export script.
    """
    st.pyplot(figure)

    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    columns = st.columns([1, 1, 3])
    with columns[0]:
        st.download_button(
            "Download SVG",
            data=buffer.getvalue(),
            file_name=f"{figure_name}.svg",
            mime="image/svg+xml",
            key=f"download_{figure_name}",
        )
    with columns[1]:
        if st.button("Write to results", key=f"write_{figure_name}"):
            FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
            path = FIGURE_ROOT / f"{figure_name}.svg"
            path.write_text(buffer.getvalue(), encoding="utf-8")
            st.success(f"wrote {path}")
