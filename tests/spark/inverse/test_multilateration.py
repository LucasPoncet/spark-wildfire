import numpy as np
import pytest

from src.config.multi_source_localization_configuration import (
    PositionRefinementConfiguration,
)
from src.spark.inverse.multilateration import (
    compute_degrees_of_freedom,
    compute_predicted_time_differences_s,
    compute_time_difference_jacobian,
    refine_position_gauss_newton,
)
from src.spark.inverse.receiver_pair_index import enumerate_receiver_pairs

SPEED_OF_SOUND_M_PER_S: float = 340.0
RECEIVER_HEIGHT_M: float = 1.5
SOURCE_HEIGHT_M: float = 0.5
REFINEMENT = PositionRefinementConfiguration(
    maximum_gauss_newton_iterations=50,
    position_convergence_tolerance_m=1e-10,
    use_band_level_terms=False,
)


def build_ring_receivers_xyz_m(receiver_count: int) -> np.ndarray:
    bearings_rad = np.linspace(0.0, 2.0 * np.pi, receiver_count, endpoint=False)
    return np.stack(
        (
            50.0 + 45.0 * np.cos(bearings_rad),
            50.0 + 45.0 * np.sin(bearings_rad),
            np.full(receiver_count, RECEIVER_HEIGHT_M),
        ),
        axis=1,
    )


def test_a_source_equidistant_from_a_pair_puts_no_delay_on_it() -> None:
    receivers_xyz_m = np.array(
        [[0.0, 0.0, RECEIVER_HEIGHT_M], [10.0, 0.0, RECEIVER_HEIGHT_M]]
    )
    delays_s = compute_predicted_time_differences_s(
        np.array([5.0, 20.0, SOURCE_HEIGHT_M]),
        receivers_xyz_m,
        enumerate_receiver_pairs(2),
        SPEED_OF_SOUND_M_PER_S,
    )
    assert delays_s == pytest.approx([0.0], abs=1e-12)


def test_a_positive_delay_means_the_first_receiver_is_the_farther_one() -> None:
    receivers_xyz_m = np.array(
        [[0.0, 0.0, RECEIVER_HEIGHT_M], [10.0, 0.0, RECEIVER_HEIGHT_M]]
    )
    delays_s = compute_predicted_time_differences_s(
        np.array([30.0, 0.0, SOURCE_HEIGHT_M]),
        receivers_xyz_m,
        enumerate_receiver_pairs(2),
        SPEED_OF_SOUND_M_PER_S,
    )
    assert delays_s[0] > 0.0


def test_the_jacobian_matches_a_finite_difference() -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m(5)
    receiver_pairs = enumerate_receiver_pairs(5)
    position_xyz_m = np.array([37.0, 62.0, SOURCE_HEIGHT_M])
    analytic = compute_time_difference_jacobian(
        position_xyz_m, receivers_xyz_m, receiver_pairs, SPEED_OF_SOUND_M_PER_S
    )
    step_m = 1e-6
    numeric = np.empty_like(analytic)
    for axis in range(2):
        offset_m = np.zeros(3)
        offset_m[axis] = step_m
        numeric[:, axis] = (
            compute_predicted_time_differences_s(
                position_xyz_m + offset_m,
                receivers_xyz_m,
                receiver_pairs,
                SPEED_OF_SOUND_M_PER_S,
            )
            - compute_predicted_time_differences_s(
                position_xyz_m - offset_m,
                receivers_xyz_m,
                receiver_pairs,
                SPEED_OF_SOUND_M_PER_S,
            )
        ) / (2.0 * step_m)
    assert np.allclose(analytic, numeric, atol=1e-12)


def test_noiseless_delays_recover_the_position_to_a_micrometre() -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m(6)
    receiver_pairs = enumerate_receiver_pairs(6)
    true_position_xyz_m = np.array([32.0, 71.0, SOURCE_HEIGHT_M])
    delays_s = compute_predicted_time_differences_s(
        true_position_xyz_m, receivers_xyz_m, receiver_pairs, SPEED_OF_SOUND_M_PER_S
    )
    estimate = refine_position_gauss_newton(
        np.array([45.0, 55.0, SOURCE_HEIGHT_M]),
        delays_s,
        np.full(delays_s.size, 1e-12),
        receivers_xyz_m,
        receiver_pairs,
        SPEED_OF_SOUND_M_PER_S,
        REFINEMENT,
    )
    assert estimate.has_converged
    assert np.allclose(estimate.position_xy_m, true_position_xyz_m[:2], atol=1e-6)
    assert estimate.reduced_chi_square == pytest.approx(0.0, abs=1e-6)


def test_the_covariance_grows_with_the_delay_variance() -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m(6)
    receiver_pairs = enumerate_receiver_pairs(6)
    true_position_xyz_m = np.array([32.0, 71.0, SOURCE_HEIGHT_M])
    delays_s = compute_predicted_time_differences_s(
        true_position_xyz_m, receivers_xyz_m, receiver_pairs, SPEED_OF_SOUND_M_PER_S
    )
    areas_m2 = [
        float(
            np.sqrt(
                np.linalg.det(
                    refine_position_gauss_newton(
                        true_position_xyz_m,
                        delays_s,
                        np.full(delays_s.size, variance_s2),
                        receivers_xyz_m,
                        receiver_pairs,
                        SPEED_OF_SOUND_M_PER_S,
                        REFINEMENT,
                    ).position_covariance_m2
                )
            )
        )
        for variance_s2 in (1e-12, 1e-10, 1e-8)
    ]
    assert areas_m2[0] < areas_m2[1] < areas_m2[2]


def test_three_receivers_are_flagged_as_ambiguous_and_five_are_not() -> None:
    for receiver_count, expected_ambiguity in ((3, True), (5, False)):
        receivers_xyz_m = build_ring_receivers_xyz_m(receiver_count)
        receiver_pairs = enumerate_receiver_pairs(receiver_count)
        true_position_xyz_m = np.array([32.0, 71.0, SOURCE_HEIGHT_M])
        estimate = refine_position_gauss_newton(
            true_position_xyz_m,
            compute_predicted_time_differences_s(
                true_position_xyz_m,
                receivers_xyz_m,
                receiver_pairs,
                SPEED_OF_SOUND_M_PER_S,
            ),
            np.full(receiver_pairs.shape[0], 1e-12),
            receivers_xyz_m,
            receiver_pairs,
            SPEED_OF_SOUND_M_PER_S,
            REFINEMENT,
        )
        assert estimate.is_ambiguous is expected_ambiguity


def test_the_residual_has_no_degrees_of_freedom_at_three_receivers() -> None:
    assert compute_degrees_of_freedom(3) == 0
    assert compute_degrees_of_freedom(4) == 1
    assert compute_degrees_of_freedom(5) == 2


def test_mismatched_delay_and_variance_lengths_are_rejected() -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m(4)
    receiver_pairs = enumerate_receiver_pairs(4)
    with pytest.raises(ValueError, match="one variance is required"):
        refine_position_gauss_newton(
            np.array([50.0, 50.0, SOURCE_HEIGHT_M]),
            np.zeros(6),
            np.zeros(5),
            receivers_xyz_m,
            receiver_pairs,
            SPEED_OF_SOUND_M_PER_S,
            REFINEMENT,
        )


def test_a_non_positive_delay_variance_is_rejected() -> None:
    receivers_xyz_m = build_ring_receivers_xyz_m(4)
    receiver_pairs = enumerate_receiver_pairs(4)
    with pytest.raises(ValueError, match="strictly positive"):
        refine_position_gauss_newton(
            np.array([50.0, 50.0, SOURCE_HEIGHT_M]),
            np.zeros(6),
            np.zeros(6),
            receivers_xyz_m,
            receiver_pairs,
            SPEED_OF_SOUND_M_PER_S,
            REFINEMENT,
        )
