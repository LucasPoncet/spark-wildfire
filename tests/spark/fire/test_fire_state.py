import dataclasses

import numpy as np
import pytest

from src.spark.fire.fire_state import FireState

CELL_COUNT = 5


def build_state() -> FireState:
    return FireState(
        ignition_times_s=np.full(CELL_COUNT, np.inf),
        burnout_times_s=np.full(CELL_COUNT, np.inf),
        is_burning=np.zeros(CELL_COUNT, dtype=np.bool_),
        has_ignited=np.zeros(CELL_COUNT, dtype=np.bool_),
        current_time_s=0.0,
    )


def test_fields_cannot_be_rebound() -> None:
    state = build_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.current_time_s = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    ["ignition_times_s", "burnout_times_s", "is_burning", "has_ignited"],
)
def test_arrays_cannot_be_mutated_in_place(field_name: str) -> None:
    state = build_state()
    with pytest.raises(ValueError, match="read-only"):
        getattr(state, field_name)[0] = 1


def test_arrays_keep_their_values() -> None:
    state = build_state()
    np.testing.assert_array_equal(state.is_burning, np.zeros(CELL_COUNT, dtype=bool))
    np.testing.assert_array_equal(state.ignition_times_s, np.inf)
    assert state.current_time_s == 0.0
