import numpy as np
import pytest

from src.audio.octave_band_filter import (
    THIRD_OCTAVE_FRACTION_DENOMINATOR,
    build_fractional_octave_centre_frequencies_hz,
    compute_fractional_octave_band_edges_hz,
)
from src.config.multi_source_localization_configuration import (
    MultiSourceLocalizationConfiguration,
    PositionRefinementConfiguration,
)
from src.spark.atmosphere.atmospheric_absorption import (
    compute_absorption_coefficients_db_per_m,
)
from src.spark.atmosphere.atmospheric_conditions import (
    AtmosphericConditions,
    compute_speed_of_sound_m_per_s,
)
from src.spark.inverse.joint_position_refinement import (
    compute_level_difference_jacobian,
    compute_pair_level_differences_db,
    compute_predicted_level_differences_db,
    refine_position_with_band_levels,
)
from src.spark.inverse.multilateration import compute_predicted_time_differences_s
from src.spark.inverse.receiver_pair_index import enumerate_receiver_pairs
from src.utils.array_types import Float64Array
from tests.spark.inverse.conftest import (
    SCENE_SOURCE_HEIGHT_M,
    render_scene_signals,
)

WINDOW_SAMPLE_COUNT: int = 8192
WINDOW_OVERLAP_FRACTION: float = 0.5
MAXIMUM_EDGE_FRACTION_OF_NYQUIST: float = 0.99


def build_bands(sample_rate_hz: int) -> tuple[Float64Array, Float64Array]:
    centres_hz = build_fractional_octave_centre_frequencies_hz(
        1000.0, 8000.0, THIRD_OCTAVE_FRACTION_DENOMINATOR
    )
    edges_hz = np.array(
        [
            compute_fractional_octave_band_edges_hz(
                centre_hz,
                THIRD_OCTAVE_FRACTION_DENOMINATOR,
                sample_rate_hz,
                MAXIMUM_EDGE_FRACTION_OF_NYQUIST,
            )
            for centre_hz in centres_hz
        ]
    )
    return np.asarray(centres_hz, dtype=np.float64), edges_hz


def test_a_source_equidistant_from_a_pair_has_no_level_difference() -> None:
    receivers_xyz_m = np.array([[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]])
    differences_db = compute_predicted_level_differences_db(
        np.array([5.0, 20.0, SCENE_SOURCE_HEIGHT_M]),
        receivers_xyz_m,
        enumerate_receiver_pairs(2),
    )
    assert differences_db == pytest.approx([0.0], abs=1e-9)


def test_the_farther_receiver_of_a_pair_hears_it_quieter() -> None:
    receivers_xyz_m = np.array([[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]])
    differences_db = compute_predicted_level_differences_db(
        np.array([30.0, 0.0, SCENE_SOURCE_HEIGHT_M]),
        receivers_xyz_m,
        enumerate_receiver_pairs(2),
    )
    assert differences_db[0] < 0.0


def test_the_level_jacobian_matches_a_finite_difference() -> None:
    receivers_xyz_m = np.array(
        [[0.0, 0.0, 1.5], [40.0, 0.0, 1.5], [20.0, 35.0, 1.5], [20.0, -35.0, 1.5]]
    )
    receiver_pairs = enumerate_receiver_pairs(4)
    position_xyz_m = np.array([17.0, 12.0, SCENE_SOURCE_HEIGHT_M])
    analytic = compute_level_difference_jacobian(
        position_xyz_m, receivers_xyz_m, receiver_pairs
    )
    step_m = 1e-6
    numeric = np.empty_like(analytic)
    for axis in range(2):
        offset_m = np.zeros(3)
        offset_m[axis] = step_m
        numeric[:, axis] = (
            compute_predicted_level_differences_db(
                position_xyz_m + offset_m, receivers_xyz_m, receiver_pairs
            )
            - compute_predicted_level_differences_db(
                position_xyz_m - offset_m, receivers_xyz_m, receiver_pairs
            )
        ) / (2.0 * step_m)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_pair_level_differences_subtract_the_second_receiver_from_the_first() -> None:
    levels_db = np.array([[10.0, 20.0], [4.0, 8.0], [1.0, 2.0]])
    variances_db2 = np.ones_like(levels_db)
    differences_db, variances = compute_pair_level_differences_db(
        levels_db, variances_db2, enumerate_receiver_pairs(3)
    )
    assert differences_db.shape == (3, 2)
    assert np.allclose(differences_db[0], [6.0, 12.0])
    assert np.all(variances > 2.0)


def test_adding_the_level_terms_shrinks_the_error_ellipse(
    scene_receivers_xyz_m: Float64Array,
    atmosphere: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
) -> None:
    """Why the level stage exists at all.

    Delays constrain range differences and nothing else, so a delay-only
    solution is loose along the direction the array cannot resolve. The
    absorption slope depends on absolute path length, which is exactly that
    direction, so the ellipse shrinks once the two are fitted together.
    """
    true_position_xyz_m = np.array([12.0, 27.0, SCENE_SOURCE_HEIGHT_M])
    receiver_signals = render_scene_signals(
        np.array(true_position_xyz_m[:2]),
        np.array([1.0]),
        scene_receivers_xyz_m,
        atmosphere,
        sample_rate_hz,
        seed=6,
    )
    receiver_pairs = enumerate_receiver_pairs(scene_receivers_xyz_m.shape[0])
    speed_of_sound_m_per_s = compute_speed_of_sound_m_per_s(
        atmosphere.air_temperature_celsius
    )
    delays_s = compute_predicted_time_differences_s(
        true_position_xyz_m,
        scene_receivers_xyz_m,
        receiver_pairs,
        speed_of_sound_m_per_s,
    )
    centres_hz, edges_hz = build_bands(sample_rate_hz)
    absorption_db_per_m = compute_absorption_coefficients_db_per_m(
        centres_hz, atmosphere
    )

    def refine(use_band_level_terms: bool) -> tuple[Float64Array, Float64Array]:
        refined = refine_position_with_band_levels(
            true_position_xyz_m,
            delays_s,
            np.full(delays_s.size, 1e-9),
            receiver_signals,
            scene_receivers_xyz_m,
            receiver_pairs,
            sample_rate_hz,
            centres_hz,
            edges_hz,
            absorption_db_per_m,
            speed_of_sound_m_per_s,
            WINDOW_SAMPLE_COUNT,
            WINDOW_OVERLAP_FRACTION,
            PositionRefinementConfiguration(
                maximum_gauss_newton_iterations=(
                    multi_source.refinement.maximum_gauss_newton_iterations
                ),
                position_convergence_tolerance_m=1e-6,
                use_band_level_terms=use_band_level_terms,
            ),
        )
        return refined.position_xy_m, refined.position_covariance_m2

    delay_only_position_xy_m, delay_only_covariance_m2 = refine(False)
    fused_position_xy_m, fused_covariance_m2 = refine(True)

    delay_only_semi_axis_m = float(
        np.sqrt(np.max(np.linalg.eigvalsh(delay_only_covariance_m2)))
    )
    fused_semi_axis_m = float(np.sqrt(np.max(np.linalg.eigvalsh(fused_covariance_m2))))
    assert fused_semi_axis_m < delay_only_semi_axis_m
    assert np.allclose(delay_only_position_xy_m, true_position_xyz_m[:2], atol=1.0)
    assert np.allclose(fused_position_xy_m, true_position_xyz_m[:2], atol=2.0)
