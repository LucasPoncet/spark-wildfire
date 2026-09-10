import numpy as np
import pytest

from src.spark.inverse.receiver_pair_index import (
    compute_maximum_absolute_lag_s,
    compute_pair_baseline_distances_m,
    enumerate_receiver_pairs,
)

SPEED_OF_SOUND_M_PER_S: float = 340.0


def test_the_pair_count_is_n_choose_two() -> None:
    for receiver_count in (2, 3, 4, 6, 8):
        pairs = enumerate_receiver_pairs(receiver_count)
        assert pairs.shape == (receiver_count * (receiver_count - 1) // 2, 2)


def test_every_pair_is_canonically_ordered() -> None:
    pairs = enumerate_receiver_pairs(5)
    assert np.all(pairs[:, 0] < pairs[:, 1])


def test_the_pair_order_is_stable_and_lexicographic() -> None:
    assert enumerate_receiver_pairs(3).tolist() == [[0, 1], [0, 2], [1, 2]]


def test_fewer_than_two_receivers_form_no_pair() -> None:
    with pytest.raises(ValueError, match="at least two receivers"):
        enumerate_receiver_pairs(1)


def test_baselines_match_the_geometry() -> None:
    positions_xyz_m = np.array(
        [[0.0, 0.0, 1.5], [3.0, 4.0, 1.5], [0.0, 4.0, 1.5]], dtype=np.float64
    )
    pairs = enumerate_receiver_pairs(3)
    baselines_m = compute_pair_baseline_distances_m(positions_xyz_m, pairs)
    assert baselines_m == pytest.approx([5.0, 4.0, 3.0])


def test_the_maximum_lag_covers_the_longest_baseline_plus_the_margin() -> None:
    positions_xyz_m = np.array(
        [[0.0, 0.0, 1.5], [100.0, 0.0, 1.5], [0.0, 50.0, 1.5]], dtype=np.float64
    )
    pairs = enumerate_receiver_pairs(3)
    longest_baseline_m = float(
        np.max(compute_pair_baseline_distances_m(positions_xyz_m, pairs))
    )
    lag_s = compute_maximum_absolute_lag_s(
        positions_xyz_m, pairs, SPEED_OF_SOUND_M_PER_S, 1.1
    )
    assert longest_baseline_m == pytest.approx(np.hypot(100.0, 50.0))
    assert lag_s == pytest.approx(1.1 * longest_baseline_m / SPEED_OF_SOUND_M_PER_S)
