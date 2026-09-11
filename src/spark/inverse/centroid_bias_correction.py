"""Undoing the centroid offset a non-symmetric array response introduces.

A response that is not symmetric about its source has a centroid displaced
from it, and the whole bearing stage reads a direction off centroid motion. An
uncorrected displacement that grows with radius is indistinguishable from a
fire spreading outward, which is the one thing the bearing stage exists to
measure.

Measured on the twenty-receiver ring of `configs/f1`, the displacement is not
the small edge effect it was expected to be. It is zero at the exact domain
centre — a symmetry point of the imaging grid, where it vanishes by
construction rather than by merit — 0.86 m only three and a half metres away,
and it peaks near 3.1 m about ten to fifteen metres out, which is precisely
where a fifty-second front sits. Past the receiver ring it reaches eleven
metres. The response is also *widest* at the centre, not narrowest: 12.1 m
semi-axis there against about 8 m at half the ring radius. Correction here is
load-bearing rather than a refinement.

The offset is tabulated once on a coarse grid and interpolated. Correction is
by subtraction, iterated: the offset is a function of where the source is, not
of where its centroid landed, so the first subtraction only moves the lookup
closer to the right place.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.array_types import Float64Array

CORRECTION_ITERATION_COUNT: int = 2


@dataclass(frozen=True)
class CentroidBiasField:
    """Measured centroid offset at each node of a regular grid.

    Attributes:
        node_positions_xy_m: Node centres, shape `(n_nodes, 2)`, with x varying
            fastest.
        offsets_xy_m: Estimated centroid minus true position at each node,
            shape `(n_nodes, 2)`.
        grid_shape: `(n_y, n_x)` of the node grid.
    """

    node_positions_xy_m: Float64Array
    offsets_xy_m: Float64Array
    grid_shape: tuple[int, int]

    @property
    def node_x_m(self) -> Float64Array:
        """Ascending unique node abscissae."""
        return np.unique(np.asarray(self.node_positions_xy_m, dtype=np.float64)[:, 0])

    @property
    def node_y_m(self) -> Float64Array:
        """Ascending unique node ordinates."""
        return np.unique(np.asarray(self.node_positions_xy_m, dtype=np.float64)[:, 1])


def save_centroid_bias_field(path: Path, field: CentroidBiasField) -> Path:
    """Writes a bias field to disk.

    Args:
        path: Destination file, created with its parents if absent.
        field: The field to write.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        node_positions_xy_m=np.asarray(field.node_positions_xy_m, dtype=np.float64),
        offsets_xy_m=np.asarray(field.offsets_xy_m, dtype=np.float64),
        grid_shape=np.asarray(field.grid_shape, dtype=np.int64),
    )
    return path


def load_centroid_bias_field(path: Path) -> CentroidBiasField:
    """Reads a bias field from disk.

    Args:
        path: File to read.

    Returns:
        The field.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"centroid bias field not found: {path}")
    with np.load(path) as document:
        grid_shape = np.asarray(document["grid_shape"], dtype=np.int64)
        return CentroidBiasField(
            node_positions_xy_m=np.asarray(
                document["node_positions_xy_m"], dtype=np.float64
            ),
            offsets_xy_m=np.asarray(document["offsets_xy_m"], dtype=np.float64),
            grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        )


def interpolate_bias_offset_xy_m(
    field: CentroidBiasField, position_xy_m: Float64Array
) -> Float64Array:
    """Bilinearly interpolates the tabulated offset at one position.

    A position outside the tabulated box is clamped to its edge rather than
    extrapolated. Extrapolating a bias that is already growing towards the
    boundary would amplify exactly where it is least trustworthy.

    Args:
        field: The tabulated field.
        position_xy_m: Where to evaluate, shape `(2,)`.

    Returns:
        The offset at that position, shape `(2,)`.
    """
    node_x_m = field.node_x_m
    node_y_m = field.node_y_m
    offsets = np.asarray(field.offsets_xy_m, dtype=np.float64).reshape(
        node_y_m.size, node_x_m.size, 2
    )
    position = np.asarray(position_xy_m, dtype=np.float64)

    column, column_weight = _locate_on_axis(node_x_m, float(position[0]))
    row, row_weight = _locate_on_axis(node_y_m, float(position[1]))
    next_column = min(column + 1, node_x_m.size - 1)
    next_row = min(row + 1, node_y_m.size - 1)

    lower = (1.0 - column_weight) * offsets[row, column] + column_weight * offsets[
        row, next_column
    ]
    upper = (1.0 - column_weight) * offsets[next_row, column] + column_weight * offsets[
        next_row, next_column
    ]
    return np.asarray((1.0 - row_weight) * lower + row_weight * upper, dtype=np.float64)


def apply_centroid_bias_correction(
    observed_centroid_xy_m: Float64Array,
    field: CentroidBiasField,
    iteration_count: int = CORRECTION_ITERATION_COUNT,
) -> Float64Array:
    """Removes the tabulated offset from an observed centroid.

    Args:
        observed_centroid_xy_m: Centroid read off a frame, shape `(2,)`.
        field: The tabulated field.
        iteration_count: How many times to re-look-up and re-subtract.

    Returns:
        The corrected position, shape `(2,)`.

    Raises:
        ValueError: If fewer than one iteration is requested.
    """
    if iteration_count < 1:
        raise ValueError("the correction needs at least one iteration")
    observed = np.asarray(observed_centroid_xy_m, dtype=np.float64)
    corrected = observed
    for _ in range(iteration_count):
        corrected = observed - interpolate_bias_offset_xy_m(field, corrected)
    return np.asarray(corrected, dtype=np.float64)


def compute_maximum_offset_magnitude_m(field: CentroidBiasField) -> float:
    """Largest offset anywhere in the field, as a summary of its severity.

    Args:
        field: The tabulated field.

    Returns:
        The largest offset magnitude, in metres.
    """
    return float(
        np.max(np.linalg.norm(np.asarray(field.offsets_xy_m, dtype=np.float64), axis=1))
    )


def has_no_sign_flip_between_adjacent_nodes(field: CentroidBiasField) -> bool:
    """Whether the field is smooth enough to interpolate rather than noisy.

    A bias that reverses direction between neighbouring nodes is not a
    geometric offset being sampled, it is estimation noise, and interpolating
    it would inject that noise into every corrected centroid.

    Args:
        field: The tabulated field.

    Returns:
        True when no component changes sign across adjacent nodes in either
        direction, ignoring changes that straddle zero by less than the
        field's own median magnitude.
    """
    row_count, column_count = field.grid_shape
    offsets = np.asarray(field.offsets_xy_m, dtype=np.float64).reshape(
        row_count, column_count, 2
    )
    magnitude_floor_m = float(np.median(np.linalg.norm(offsets.reshape(-1, 2), axis=1)))
    for axis in (0, 1):
        differences = np.diff(offsets, axis=axis)
        leading = np.take(offsets, np.arange(offsets.shape[axis] - 1), axis=axis)
        flips = (np.sign(leading) != np.sign(leading + differences)) & (
            np.abs(leading) > magnitude_floor_m
        )
        if bool(np.any(flips)):
            return False
    return True


def _locate_on_axis(axis_values_m: Float64Array, value_m: float) -> tuple[int, float]:
    """Finds the cell a value falls in and how far across it sits.

    Args:
        axis_values_m: Ascending node coordinates along one axis.
        value_m: The coordinate to locate.

    Returns:
        `(lower_index, weight)` with the weight in `[0, 1]`.
    """
    values = np.asarray(axis_values_m, dtype=np.float64)
    if values.size == 1:
        return 0, 0.0
    lower_index = int(np.clip(np.searchsorted(values, value_m) - 1, 0, values.size - 2))
    span_m = float(values[lower_index + 1] - values[lower_index])
    if span_m <= 0.0:
        return lower_index, 0.0
    weight = (value_m - float(values[lower_index])) / span_m
    return lower_index, float(np.clip(weight, 0.0, 1.0))
