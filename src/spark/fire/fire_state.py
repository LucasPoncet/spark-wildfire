"""Immutable snapshot of the fire simulation at one instant in time."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class FireState:
    """State of every cell at `current_time_s`. `step` returns a new instance.

    Attributes:
        ignition_times_s: Float64 array of shape (n,). Time each cell caught
            fire, in seconds. Undefined (left at its previous value) for
            cells that have never ignited.
        burnout_times_s: Float64 array of shape (n,). Time each cell is due
            to finish burning, in seconds. Undefined for cells that have
            never ignited.
        is_burning: Bool array of shape (n,). True while a cell is actively
            on fire.
        has_ignited: Bool array of shape (n,). True from ignition onward,
            including after burnout — a cell that has burnt can never
            ignite again.
        current_time_s: Simulation time this snapshot corresponds to.
    """

    ignition_times_s: npt.NDArray[np.float64]
    burnout_times_s: npt.NDArray[np.float64]
    is_burning: npt.NDArray[np.bool_]
    has_ignited: npt.NDArray[np.bool_]
    current_time_s: float
