"""One definition of receiver pair ordering, for the whole inverse package.

Every delay in this package is signed, and the sign only means something against
a stated ordering. Pairs are canonical `(i, j)` with `i < j`, and a positive
delay means receiver `i` hears the event later than receiver `j`.
"""

import numpy as np

from src.utils.array_types import Float64Array, Int64Array


def enumerate_receiver_pairs(receiver_count: int) -> Int64Array:
    """Lists every unordered receiver pair in canonical order.

    Args:
        receiver_count: Number of receivers.

    Returns:
        Pairs of shape `(receiver_count * (receiver_count - 1) / 2, 2)`, each row
        `(i, j)` with `i < j`, ordered by `i` then `j`.

    Raises:
        ValueError: If fewer than two receivers are given.
    """
    if receiver_count < 2:
        raise ValueError("at least two receivers are required to form a pair")
    first_index, second_index = np.triu_indices(receiver_count, k=1)
    return np.stack((first_index, second_index), axis=1).astype(np.int64)


def compute_pair_baseline_distances_m(
    receiver_positions_xyz_m: Float64Array, receiver_pairs: Int64Array
) -> Float64Array:
    """Measures the baseline of every receiver pair.

    Args:
        receiver_positions_xyz_m: Positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.

    Returns:
        Baseline length per pair, shape `(n_pairs,)`, in metres.
    """
    positions = np.atleast_2d(np.asarray(receiver_positions_xyz_m, dtype=np.float64))
    pairs = np.asarray(receiver_pairs, dtype=np.int64)
    return np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)


def compute_maximum_absolute_lag_s(
    receiver_positions_xyz_m: Float64Array,
    receiver_pairs: Int64Array,
    speed_of_sound_m_per_s: float,
    lag_margin: float,
) -> float:
    """Computes how far a correlation curve has to be searched.

    No source can put a delay on a pair larger than the pair's baseline divided
    by the speed of sound, so the curve is retained only that far out, plus a
    margin for an imperfect assumed speed of sound.

    Args:
        receiver_positions_xyz_m: Positions, shape `(n_receivers, 3)`.
        receiver_pairs: Canonical pairs, shape `(n_pairs, 2)`.
        speed_of_sound_m_per_s: Assumed speed of sound.
        lag_margin: Multiplier on the geometric limit.

    Returns:
        Largest absolute lag to retain, in seconds.
    """
    baselines_m = compute_pair_baseline_distances_m(
        receiver_positions_xyz_m, receiver_pairs
    )
    return float(lag_margin * np.max(baselines_m) / speed_of_sound_m_per_s)
